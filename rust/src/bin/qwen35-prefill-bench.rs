//! Times Qwen3.5 prompt prefill (one forward pass over N tokens) with Candle.
//!
//! Needs only a public Qwen3.5 checkpoint, for example `Qwen/Qwen3.5-4B`; nothing here is
//! specific to Wisp. Prints, for each prompt length, the median and minimum wall time and a
//! checksum of the last-position logits, so two builds or two environment switches can be
//! compared for both speed and the same answer.
//!
//!   qwen35-prefill-bench --model-dir <checkpoint dir> --device cuda --tokens 64,256,1024
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_5::{Config, ModelForCausalLM, TextConfig};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
}

#[derive(Debug, Parser)]
#[command(about = "Qwen3.5 prompt-prefill timing with Candle")]
struct Args {
    /// Directory with config.json and the safetensors weights.
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, value_enum, default_value = "cuda")]
    device: DeviceArg,
    /// Prompt lengths to time, comma separated.
    #[arg(long, value_delimiter = ',', default_value = "64,256,1024")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
}

fn weight_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&index)?)?;
    let map = value["weight_map"].as_object().context("weight_map")?;
    let mut names: Vec<String> = map.values().filter_map(|v| v.as_str().map(str::to_owned)).collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device {
        DeviceArg::Cpu => Device::Cpu,
        DeviceArg::Cuda => Device::new_cuda(0)?,
    };
    let config_value: serde_json::Value = serde_json::from_slice(&fs::read(args.model_dir.join("config.json"))?)?;
    let config: Config = if config_value.get("text_config").is_some() {
        serde_json::from_value(config_value)?
    } else {
        Config { text_config: serde_json::from_value::<TextConfig>(config_value)? }
    };
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let files = weight_files(&args.model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let mut model = ModelForCausalLM::new(&config, vb).context("load Qwen3.5")?;

    let vocab = config.text_config.vocab_size as u32;
    println!("device={:?} dtype={dtype:?} repeats={}", args.device, args.repeats);
    println!("{:>7} {:>10} {:>10}  checksum (argmax, sum of the 16 largest logits)", "tokens", "median_ms", "min_ms");
    for &n in &args.tokens {
        if n == 0 {
            bail!("--tokens must be positive");
        }
        // Fixed pseudo-text: prefill cost does not depend on the token values.
        let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % (vocab.min(100_000) - 1000)).collect();
        let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
        let mut times = Vec::new();
        let mut checksum = (0usize, 0f32);
        for round in 0..=args.repeats {
            model.clear_kv_cache();
            let start = Instant::now();
            let logits = model.forward(&input, 0)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let values = logits.to_vec1::<f32>()?; // also waits for the GPU
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if round > 0 {
                times.push(ms); // round 0 warms up kernels and allocations
            }
            let argmax = values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            checksum = (argmax, sorted.iter().take(16).sum());
        }
        times.sort_by(|a, b| a.total_cmp(b));
        println!("{n:>7} {:>10.1} {:>10.1}  ({}, {:.3})", times[times.len() / 2], times[0], checksum.0, checksum.1);
    }
    Ok(())
}
