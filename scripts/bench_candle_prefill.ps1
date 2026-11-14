# Times Qwen3.5 prompt prefill with Candle on the GPU (see rust/src/bin/qwen35-prefill-bench.rs).
# Stop any running Wisp server first: the checkpoint is loaded in BF16 (about 9 GB for Qwen3.5-4B).
param(
    [Parameter(Mandatory)] [string]$ModelDir,
    [string]$Tokens = "64,256,1024",
    [int]$Repeats = 5
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\candle_cuda_env.ps1"

& cargo build --release --manifest-path "$repo\rust\Cargo.toml" --features $features --bin qwen35-prefill-bench
if ($LASTEXITCODE -ne 0) { throw "Rust/Candle build failed with exit code $LASTEXITCODE" }
& "$repo\rust\target\release\qwen35-prefill-bench.exe" --model-dir $ModelDir --device cuda --tokens $Tokens --repeats $Repeats
