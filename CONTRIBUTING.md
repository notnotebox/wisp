# Contributing

Wisp v0.1.2 is an experimental release with a trained model, a working runtime, and a fixed evaluation baseline. Contributions that improve its coverage, calibration, and portability are welcome.

Issues and pull requests are welcome, especially bug reports with the request that triggered them.

- **Python package** (`packages/wisp-decide`): `PYTHONPATH=packages/wisp-decide/src python -m unittest discover -s packages/wisp-decide/tests`. The tests use a fake model, so no download or GPU is needed.
- **Rust runtime** (`rust/`): `cargo test --release --features inference --manifest-path rust/Cargo.toml` runs the CPU tests. The CUDA kernel tests need an NVIDIA GPU (`scripts/test_candle_cuda.ps1`, Windows).
- **Vendored Candle** (`rust/vendor/candle`): keep local changes small and listed in `rust/vendor/candle/WISP_VENDOR.md`.
- Please do not include model weights, API keys or private data in issues or pull requests.

By contributing you agree that your contribution is licensed under the repository's licence (Apache-2.0).
