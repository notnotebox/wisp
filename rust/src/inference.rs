//! Native Candle inference for the merged Wisp v0.1.2 4B checkpoint.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_5::{Config, ModelForCausalLM, TextConfig};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const SYSTEM: &str = "You are Wisp, a moderation decision engine. You receive a STATE (a chat message with optional history, a report, or a username), sometimes a CONTEXT (server policy or rules), a QUESTION and OPTIONS, each with a symbol. Choose the single option that best answers the QUESTION for the STATE, following the CONTEXT when one is given. Return only the symbol of that option.";
const USERNAME_POLICY: &str = "USERNAME POLICY: REVIEW clearly insulting or profane player names, including recognizable abbreviations and spelling evasions, even without a named target. Also REVIEW explicit extremist praise, targeted harassment, sexual solicitation, doxxing, scams or clear hateful slurs. NO_ACTION for neutral historical references, ordinary game words, anti-extremist names and harmless lookalikes. Judge the whole name, not a substring. NEED_CONTEXT when the meaning truly cannot be inferred. REVIEW means human inspection, never an automatic ban.";
const CHAT_POLICY: &str = "CHAT POLICY: REVIEW a directed insult, harassment, credible threat, hate, doxxing, sexual exploitation, scam or serious abuse, including clear spelling evasions. NO_ACTION for a victim quoting abuse to report it, discussion of filtering, harmless gameplay violence, ordinary disagreement or neutral language. NEED_CONTEXT when the message is genuinely uninterpretable without more context. REVIEW never imposes an automatic sanction.";
const CATEGORY_POLICY: &str = "CLASSIFICATION POLICY: INSULT means a personal insult or direct verbal abuse; THREAT means credible threat; HARASSMENT means repeated targeted abuse; HATE means attack on a protected group; SEXUAL means sexual harassment or exploitation; DOXXING means exposure of private information; OTHER means a different violation; NONE means no violation by the author of the supplied text. Classify the actual message or username, not merely a mentioned word.";
const REPORT_POLICY: &str = "REPORT POLICY: VALID_REPORT when the supplied text shows a plausible policy violation needing moderator review. INVALID_REPORT when the claim is unsupported, retaliatory or concerns allowed behavior. The report outcome is triage, not guilt or an automatic sanction.";
const GENERIC_POLICY: &str = "Answer the supplied classification question about the STATE. Use only the offered options.";

#[derive(Debug, Clone, Deserialize)]
pub struct DecideRequest {
    pub task: Option<String>,
    pub text: String,
    pub question: String,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecideResult {
    pub decision: String,
    pub probabilities: std::collections::BTreeMap<String, f64>,
    pub confidence: f64,
    pub margin: f64,
    pub needs_review: bool,
    pub mapping_disagreement: u8,
    pub calibration: &'static str,
    pub model: &'static str,
    pub revision: String,
    pub runtime: &'static str,
    pub latency_ms: u128,
}

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: std::collections::HashMap<String, String>,
}

pub struct WispEngine {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    choice_token_ids: Vec<u32>,
    device: Device,
    revision: String,
}

impl WispEngine {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let config_bytes = fs::read(model_dir.join("config.json")).context("read config.json")?;
        let config_value: serde_json::Value =
            serde_json::from_slice(&config_bytes).context("parse config.json")?;
        let config: Config = if config_value.get("text_config").is_some() {
            serde_json::from_value(config_value).context("parse nested Qwen3.5 config")?
        } else {
            Config {
                text_config: serde_json::from_value::<TextConfig>(config_value)
                    .context("parse text-only Qwen3.5 config")?,
            }
        };
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let filenames = weight_files(model_dir)?;
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = ModelForCausalLM::new(&config, vb).context("load Qwen3.5 model")?;

        let mut choice_token_ids = Vec::with_capacity(SYMBOLS.len());
        for symbol in SYMBOLS.chars() {
            let encoding = tokenizer
                .encode(symbol.to_string(), false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            if encoding.get_ids().len() != 1 {
                bail!("choice symbol {symbol:?} is not one tokenizer token");
            }
            choice_token_ids.push(encoding.get_ids()[0]);
        }
        let revision = fs::read_to_string(model_dir.join("wisp_revision.txt"))
            .unwrap_or_else(|_| "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into())
            .trim()
            .to_owned();

        Ok(Self {
            model,
            tokenizer,
            choice_token_ids,
            device,
            revision,
        })
    }

    pub fn decide(&mut self, request: DecideRequest) -> Result<DecideResult> {
        validate_request(&request)?;
        let started = Instant::now();
        let prompt = render_prompt(&request);
        let ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.len() > 4096 {
            bail!("request exceeds model context limit");
        }

        self.model.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, 0)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let option_ids = &self.choice_token_ids[..request.options.len()];
        let option_ids = Tensor::new(option_ids, &self.device)?;
        let scores = logits.index_select(&option_ids, 0)?.to_vec1::<f32>()?;
        let probabilities = stable_softmax(&scores);
        let mut ranked = probabilities.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));
        let best = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .context("empty probability vector")?;
        let probabilities = request
            .options
            .iter()
            .cloned()
            .zip(probabilities)
            .collect();

        Ok(DecideResult {
            decision: request.options[best].clone(),
            probabilities,
            confidence: ranked[0],
            margin: ranked[0] - ranked[1],
            needs_review: ranked[0] < 0.7,
            mapping_disagreement: 0,
            calibration: "uncalibrated",
            model: "Qwen/Qwen3.5-4B",
            revision: self.revision.clone(),
            runtime: "candle-rust",
            latency_ms: started.elapsed().as_millis(),
        })
    }
}

fn weight_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: WeightIndex = serde_json::from_slice(&fs::read(index_path)?)?;
        let names: BTreeSet<_> = index.weight_map.into_values().collect();
        return Ok(names.into_iter().map(|name| model_dir.join(name)).collect());
    }
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    bail!("no safetensors weights found in {}", model_dir.display())
}

fn validate_request(request: &DecideRequest) -> Result<()> {
    match request.task.as_deref() {
        None | Some("moderation" | "username" | "report_validity") => {}
        Some(value) => bail!("unsupported task: {value}"),
    }
    if request.text.trim().is_empty() || request.text.chars().count() > 4000 {
        bail!("text must be 1..4000 characters");
    }
    if request.question.trim().is_empty() || request.question.chars().count() > 200 {
        bail!("question must be 1..200 characters");
    }
    if !(2..=12).contains(&request.options.len()) {
        bail!("options must contain 2..12 distinct short strings");
    }
    let mut unique = BTreeSet::new();
    for option in &request.options {
        if option.trim().is_empty() || option.chars().count() > 60 || !unique.insert(option) {
            bail!("options must contain 2..12 distinct short strings");
        }
    }
    Ok(())
}

fn render_prompt(request: &DecideRequest) -> String {
    let category_task = request
        .options
        .iter()
        .any(|o| matches!(o.as_str(), "INSULT" | "THREAT" | "HARASSMENT" | "HATE"));
    let mut policy = if category_task {
        CATEGORY_POLICY.to_owned()
    } else {
        match request.task.as_deref() {
            Some("username") => USERNAME_POLICY.to_owned(),
            Some("report_validity") => REPORT_POLICY.to_owned(),
            Some("moderation") => CHAT_POLICY.to_owned(),
            _ => GENERIC_POLICY.to_owned(),
        }
    };
    let kind = match request.task.as_deref() {
        Some("username") => "USERNAME",
        Some("report_validity") => "REPORT",
        Some("moderation") => "REPORTED MESSAGE",
        _ => "USER MESSAGE",
    };
    let options = request
        .options
        .iter()
        .enumerate()
        .map(|(index, option)| format!("{}. {option}", SYMBOLS.as_bytes()[index] as char))
        .collect::<Vec<_>>()
        .join("\n");
    let user = format!(
        "CONTEXT\n{policy}\n\nSTATE\n{kind}\n{}\n\nQUESTION\n{}\n\nOPTIONS\n{options}",
        request.text, request.question
    );
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn stable_softmax(scores: &[f32]) -> Vec<f64> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = scores.iter().map(|v| ((*v - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(options: &[&str]) -> DecideRequest {
        DecideRequest {
            task: Some("moderation".into()),
            text: "ordinary test message".into(),
            question: "Review?".into(),
            options: options.iter().map(|v| (*v).to_owned()).collect(),
        }
    }

    #[test]
    fn validates_bounded_options() {
        assert!(validate_request(&request(&["REVIEW", "NO_ACTION"])).is_ok());
        assert!(validate_request(&request(&["REVIEW"])).is_err());
        assert!(validate_request(&request(&["REVIEW", "REVIEW"])).is_err());
    }

    #[test]
    fn softmax_is_normalized() {
        let probabilities = stable_softmax(&[1.0, 2.0, 3.0]);
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(probabilities[2] > probabilities[1]);
    }

    #[test]
    fn prompt_matches_qwen_no_think_shape() {
        let rendered = render_prompt(&request(&["REVIEW", "NO_ACTION"]));
        assert!(rendered.starts_with("<|im_start|>system\n"));
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(rendered.contains("A. REVIEW\nB. NO_ACTION"));
    }
}

