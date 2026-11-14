# Wisp Candle vendor baseline

This note documents the Candle baseline and local patches used by the experimental Wisp v0.1.2 runtime.

This directory is pinned from the Qwen 3.5 Candle implementation at commit
`8aac04c428744b1d203e7d58ef6eb7f2c6297f58` (the implementation supplied by
the open upstream pull request at the time Wisp was built).

Wisp carries a deliberately small local CUDA compatibility patch in
`candle-transformers/src/models/qwen3_5.rs`: CUDA lacks Candle's generic
`sigmoid` kernel on the supported Windows backend, so the Qwen 3.5 path
expresses sigmoid and SiLU in terms of CUDA-supported primitive operations.
The algebra is exact: `sigmoid(x) = 1 / (1 + exp(-x))` and
`silu(x) = x * sigmoid(x)`.

Keep this note and the patch when updating the upstream baseline; replace the
vendor only after the parity suite passes.

## Prefill speed patches (Wisp, 2026-11-05)

The hardware and timings below describe one local kernel-development benchmark. They
are included to make that measurement reproducible; they are not minimum system
requirements or a recommendation for people running Wisp. See the model page for
runtime and download information.

Further local changes, all exact up to float rounding:

- The linear-attention layers' depthwise convolution runs as `kernel` broadcast
  multiply-adds (`depthwise_conv1d`) instead of Candle's grouped `conv1d`, which
  launches one convolution per channel (8192 per layer on Qwen3.5-4B).
  A related general Candle optimization is proposed in [upstream PR #3531](https://github.com/huggingface/candle/pull/3531); that PR is still open and is not the same as Wisp's local Qwen-specific implementation.
- Prompt prefill (`seq_len > 1`) uses the chunked GatedDeltaNet scan from
  `qwen3_5_linear_attn_scan.rs` instead of the per-token recurrence.

- `silu` uses Candle's core `Tensor::silu` (one CUDA kernel) instead of being composed
  from the sigmoid replacement (five launches). Used by the MLP and the gated norm.
- On CUDA, prompt prefill runs the gated delta rule in one fused kernel
  (`candle-kernels/src/delta_rule.cu`, `gated_delta_rule_fused_prefill`): one block per
  (batch, head), one thread per value column, the state kept in registers for the whole
  prompt. It supports `head_k_dim == head_v_dim <= 128` and prompts of up to 1024 tokens
  (`FUSED_PREFILL_MAX_TOKENS`: it was 2x to 3.7x faster than the chunked scan from 64 to 512
  tokens and 1.8x at 1024 on Qwen3.5-4B; above that, the fused kernel is not used. Heavy GPU memory pressure can also invalidate timing measurements.)
  Anything else, and non-CUDA devices, use the chunked scan above. Written for this project; it is not the kernel in upstream pull request 3890, which targets fixed Qwen3.5-4B shapes and ships no Rust glue.

`WISP_CANDLE_GROUPED_CONV=1`, `WISP_CANDLE_SEQUENTIAL_SCAN=1` and
`WISP_CANDLE_COMPOSED_SILU=1` restore the original paths, and
`WISP_CANDLE_NO_FUSED_PREFILL=1` falls back from the fused kernel to the chunked scan, for
parity and speed checks. `rust/tests/delta_rule_scan.rs` checks the convolution and the scan
against the original computations on CPU, and `rust/tests/cuda_delta_prefill.rs`
(`scripts/test_candle_cuda.ps1`) checks the fused kernel against the per-token loop on the GPU. `WISP_CANDLE_PROFILE=1` makes each
decoder section wait for the GPU and report its own time (`take_profile`).

Measured on Windows 11, CUDA 13.0, RTX 4070 SUPER, Qwen3.5-4B merged BF16, eight
short moderation requests, same binary and weights, quiet GPU:
original paths 9.8 s median, with the first three changes 0.21 s median (46x), same
decisions on all eight, largest probability difference 0.02. With the fused prefill
kernel on top: 0.10 s median (2.2x again), same decisions, largest difference 0.04.
The Python runtime (4-bit base) measured 0.13 s median in the same session.
