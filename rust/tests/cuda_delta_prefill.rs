//! The fused CUDA prefill kernel must match the per-token recurrence (run on the CPU).
#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::qwen3_5_linear_attn_scan::{
    fused_prefill_supported, gated_delta_rule_fused_prefill, sequential_step,
};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(D::Minus1).unwrap().to_scalar::<f32>().unwrap()
}

fn per_token(q: &Tensor, k: &Tensor, v: &Tensor, log_g: &Tensor, beta: &Tensor, state: &mut Tensor) -> Tensor {
    let t = q.dim(1).unwrap();
    let g = log_g.exp().unwrap();
    let outputs: Vec<Tensor> = (0..t)
        .map(|i| {
            let at = |x: &Tensor| x.narrow(1, i, 1).unwrap().squeeze(1).unwrap().contiguous().unwrap();
            sequential_step(&at(q), &at(k), &at(v), &at(&g), &at(beta), state).unwrap().unsqueeze(1).unwrap()
        })
        .collect();
    Tensor::cat(&outputs, 1).unwrap()
}

fn check(b: usize, t: usize, h: usize, d: usize) {
    let cpu = Device::Cpu;
    let gpu = Device::new_cuda(0).expect("a CUDA device is required for this test");
    let unit = |shape: (usize, usize, usize, usize)| {
        let x = Tensor::randn(0f32, 1f32, shape, &cpu).unwrap();
        let norm = x.sqr().unwrap().sum_keepdim(D::Minus1).unwrap().sqrt().unwrap();
        x.broadcast_div(&norm).unwrap()
    };
    let q = (unit((b, t, h, d)) * (1.0 / (d as f64).sqrt())).unwrap();
    let k = unit((b, t, h, d));
    let v = Tensor::randn(0f32, 1f32, (b, t, h, d), &cpu).unwrap();
    let log_g = (Tensor::rand(0f32, 1f32, (b, t, h), &cpu).unwrap() * -0.5).unwrap();
    let beta = Tensor::rand(0f32, 1f32, (b, t, h), &cpu).unwrap();
    let initial = (Tensor::randn(0f32, 1f32, (b, h, d, d), &cpu).unwrap() * 0.1).unwrap();

    let mut ref_state = initial.clone();
    let reference = per_token(&q, &k, &v, &log_g, &beta, &mut ref_state);

    let on = |x: &Tensor| x.to_device(&gpu).unwrap();
    let (qg, kg, vg) = (on(&q), on(&k), on(&v));
    assert!(fused_prefill_supported(&qg, &vg));
    let mut state = on(&initial);
    let out = gated_delta_rule_fused_prefill(&qg, &kg, &vg, &on(&log_g), &on(&beta), &mut state).unwrap();

    assert_eq!(out.dims(), reference.dims());
    assert_eq!(out.dtype(), DType::F32);
    let out_diff = max_abs_diff(&out.to_device(&cpu).unwrap(), &reference);
    let state_diff = max_abs_diff(&state.to_device(&cpu).unwrap(), &ref_state);
    assert!(out_diff < 1e-3, "b={b} t={t} h={h} d={d}: output mismatch {out_diff}");
    assert!(state_diff < 1e-3, "b={b} t={t} h={h} d={d}: state mismatch {state_diff}");
}

#[test]
fn fused_prefill_matches_per_token_loop() {
    check(1, 1, 2, 128); // a single token
    check(1, 181, 4, 128); // a typical prompt
    check(2, 70, 3, 64); // batch of 2 and a smaller head dimension
}
