# Native Rust/Candle runtime

This is the native runtime for the experimental Wisp v0.1.2 release. It is functional on the tested setup; broader platform support and runtime parity remain active areas of work.

`rust/` is a single Rust server that runs Wisp v0.1.2 4B without Python at inference time. PyTorch and Transformers remain the training and reference environment; the Python package [`wisp-decide`](../packages/wisp-decide) is a secondary option for machines where this server cannot run.

```text
client
   -> HTTP JSON
wisp-server (Rust, axum)
   -> one dedicated inference thread
Candle Qwen3.5-4B + merged Wisp weights (BF16)
   -> option logits and probabilities
```

It never generates free text. It scores the one-token symbols attached to the caller's allowed options and returns a decision, probabilities, confidence and margin. Scores are not calibrated.

## Why the Rust server needed speed work

The pinned Qwen3.5 implementation is not part of Candle `main`. Profiling the unoptimized pinned tree pointed to grouped `conv1d` in the linear-attention layers and a prompt scan that advances one token at a time. Wisp adds a Qwen-specific depthwise-convolution path, uses the chunked prompt scan, and adds a fused CUDA prefill kernel for prompts up to 1,024 tokens. The local speed check documented below shows the effect. These are measurements of this pinned implementation and setup, not a general Candle performance claim. A related general depthwise-convolution optimization is proposed in [Candle PR #3531](https://github.com/huggingface/candle/pull/3531); it remains open, and it is not the same patch as Wisp's local Qwen-specific path. Treat Wisp's changes as local maintenance; reassess them when the upstream implementation changes.

## Routes

- `POST /decide`: arbitrary bounded options (2 to 12), same body as the Python server.
- `POST /v1/moderate` and `POST /v1/username`: a review decision plus a violation category for a chat message or a username.
- `GET /v1/models`: describes the loaded model (used by the Wisp Benchmark demo).
- `GET /health`.

## Technology

- [Rust](https://www.rust-lang.org), [axum](https://github.com/tokio-rs/axum), [tokio](https://tokio.rs), [tokenizers](https://github.com/huggingface/tokenizers).
- [Candle](https://github.com/huggingface/candle) for tensors and CUDA. Qwen3.5 support is not in Candle `main`, so `rust/vendor/candle` is a copy pinned to commit `8aac04c428744b1d203e7d58ef6eb7f2c6297f58` of an open Qwen3.5 implementation. It carries a few local patches (listed in `rust/vendor/candle/WISP_VENDOR.md`): a sigmoid written with supported primitives because CUDA lacks the generic kernel on the Windows backend, and the prefill speed-ups below (exact up to float rounding). Revisit this dependency once Qwen3.5 lands upstream. Candle keeps its own Apache-2.0 and MIT licences.
- The adapter is merged once into BF16 safetensors by `scripts/merge_adapter_for_candle.py` (PyTorch, Transformers, PEFT). This does not retrain or alter the adapter.

## Build and run (Windows, CUDA)

```powershell
hf download notebox/Wisp-v0.1.2-4B --local-dir models\wisp-v0.1.2-4b
.\scripts\prepare_candle.ps1      # merges the adapter and builds rust\target\release\wisp-server.exe
.\scripts\start_candle.ps1        # serves on http://127.0.0.1:8782
```

`prepare_candle.ps1 -CpuOnly` builds a CPU-only diagnostic binary. The scripts expect the CUDA toolkit and the Visual Studio C++ build tools, and fall back to `CUDA_PATH` when no local toolkit is present. Memory use varies with the device, available memory and other running applications; no universal minimum is claimed. See the [Hugging Face model page](https://huggingface.co/notebox/Wisp-v0.1.2-4B) for release information.

## Status and known limits

- In a fixed check on 8 short cases (moderation, username and report validity, run 2026-11-08 on the public code), Python and Candle selected the same option on 7. The largest score difference was 0.34. This is decision agreement on a small sample, not numerical parity or calibration: Python uses a 4-bit NF4 base while Candle uses merged BF16 weights.
- Speed, base Qwen3.5-4B in BF16, one forward pass over N tokens, idle GPU (Windows/CUDA, median of 3):

  | Tokens | Candle (this repository) | Python (Transformers) | Faster |
  |---:|---:|---:|---|
  | 64 | 54 ms | 97 ms | Candle, about 1.8x |
  | 256 | 83 ms | 118 ms | Candle, about 1.4x |
  | 512 | 156 ms | 120 ms | Python, about 1.3x |
  | 1024 | 332 ms | 241 ms | Python, about 1.4x |

  On Wisp's own short requests (8 cases, idle GPU, one request at a time), Candle with the BF16 merged model took 83 ms on average (median 79 ms) and the Python package with a 4-bit base took 136 ms (median 142 ms). Both selected the same option on 7 of 8 cases. The per-case results are in [`measurements/rust_vs_python_8_cases_2026-11-12.json`](measurements/rust_vs_python_8_cases_2026-11-12.json). Eight requests are a small sample, and the two runtimes use different precision, so read this as an indication, not a benchmark.

  Rust is faster on short prompts (up to about 300 tokens), which is Wisp's typical request size; Python was faster on longer prompts in this specific base-model test. These are local measurements, not hardware recommendations or a throughput guarantee. See `rust/vendor/candle/WISP_VENDOR.md` for optimization details.
- Windows with CUDA is the only tested environment so far. Other operating systems and hardware configurations are unvalidated; actual memory requirements vary by setup.
