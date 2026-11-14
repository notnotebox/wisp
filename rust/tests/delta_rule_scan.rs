//! The chunked GatedDeltaNet scan used for prompt prefill must match the per-token recurrence.
#![cfg(feature = "inference")]

use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::qwen3_5_linear_attn_scan::{gated_delta_rule_chunked, sequential_step};

fn per_token(q: &Tensor, k: &Tensor, v: &Tensor, log_g: &Tensor, beta: &Tensor, state: &mut Tensor) -> Tensor {
    let t = q.dim(1).unwrap();
    let g = log_g.exp().unwrap();
    let outputs: Vec<Tensor> = (0..t)
        .map(|i| {
            let at = |x: &Tensor| x.narrow(1, i, 1).unwrap().squeeze(1).unwrap().contiguous().unwrap();
            sequential_step(&at(q), &at(k), &at(v), &at(&g), &at(beta), state).unwrap().unsqueeze(1).unwrap()
        })
        .collect();
    Tensor::cat(&outputs, 1).unwrap() // [b, t, h, v]
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(D::Minus1).unwrap().to_scalar::<f32>().unwrap()
}

#[test]
fn chunked_scan_matches_per_token_loop() {
    let dev = Device::Cpu;
    // Realistic head sizes; 150 tokens covers several chunks of 64 and a padded last chunk.
    let (b, t, h, d) = (1, 150, 4, 128);
    let unit = |shape: (usize, usize, usize, usize)| {
        let x = Tensor::randn(0f32, 1f32, shape, &dev).unwrap();
        let norm = x.sqr().unwrap().sum_keepdim(D::Minus1).unwrap().sqrt().unwrap();
        x.broadcast_div(&norm).unwrap() // the layer L2-normalises q and k
    };
    let scale = 1.0 / (d as f64).sqrt();
    let q = (unit((b, t, h, d)) * scale).unwrap();
    let k = unit((b, t, h, d));
    let v = Tensor::randn(0f32, 1f32, (b, t, h, d), &dev).unwrap();
    // Decay in (0.6, 1) as in the model, kept as its log like the layer passes it.
    let log_g = (Tensor::rand(0f32, 1f32, (b, t, h), &dev).unwrap() * -0.5).unwrap();
    let beta = Tensor::rand(0f32, 1f32, (b, t, h), &dev).unwrap();
    let initial = (Tensor::randn(0f32, 1f32, (b, h, d, d), &dev).unwrap() * 0.1).unwrap();

    let mut chunk_state = initial.clone();
    let chunked = gated_delta_rule_chunked(&q, &k, &v, &log_g, &beta, &mut chunk_state).unwrap();
    let mut loop_state = initial;
    let reference = per_token(&q, &k, &v, &log_g, &beta, &mut loop_state);

    assert_eq!(chunked.dims(), reference.dims());
    assert_eq!(chunked.dtype(), DType::F32);
    let out = max_abs_diff(&chunked, &reference);
    let state = max_abs_diff(&chunk_state, &loop_state);
    assert!(out < 1e-3, "output mismatch: {out}");
    assert!(state < 1e-3, "state mismatch: {state}");
}

#[test]
fn depthwise_conv_matches_grouped_conv1d() {
    let dev = Device::Cpu;
    // Qwen3.5-4B linear attention: 8192 channels, kernel 4; 37 tokens plus 3 of padding.
    let (b, channels, kernel, len) = (1, 8192, 4, 37);
    let padded = Tensor::randn(0f32, 1f32, (b, channels, len + kernel - 1), &dev).unwrap();
    let weight = Tensor::randn(0f32, 1f32, (channels, 1, kernel), &dev).unwrap();
    let grouped = padded.conv1d(&weight, 0, 1, 1, channels).unwrap();
    let fast = candle_transformers::models::qwen3_5::depthwise_conv1d(&padded, &weight).unwrap();
    assert_eq!(fast.dims(), grouped.dims());
    let diff = max_abs_diff(&fast, &grouped);
    assert!(diff < 1e-4, "depthwise conv mismatch: {diff}");
    // Decode step: the cached window of `kernel` positions gives one output.
    let window = padded.narrow(2, 0, kernel).unwrap();
    let diff = max_abs_diff(
        &candle_transformers::models::qwen3_5::depthwise_conv1d(&window, &weight).unwrap(),
        &window.conv1d(&weight, 0, 1, 1, channels).unwrap(),
    );
    assert!(diff < 1e-4, "decode conv mismatch: {diff}");
}
