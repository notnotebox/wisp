# Runs the CUDA tests of the Rust/Candle runtime (needs a CUDA GPU, the CUDA toolkit and MSVC).
param(
    [string]$Test = "cuda_delta_prefill"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\candle_cuda_env.ps1"

& cargo test --release --manifest-path "$repo\rust\Cargo.toml" --features $features --test $Test -- --nocapture
if ($LASTEXITCODE -ne 0) { throw "Rust/Candle CUDA test failed with exit code $LASTEXITCODE" }
