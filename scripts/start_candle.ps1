param(
    [int]$Port = 8782,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$model = Join-Path $repo "models\wisp-v0.1.2-4b-candle"
$binary = Join-Path $repo "rust\target\release\wisp-server.exe"
if (-not (Test-Path (Join-Path $model "wisp_candle_manifest.json"))) { throw "Merged model is missing. Run scripts\prepare_candle.ps1 first." }
if (-not (Test-Path $binary)) { throw "Rust server is missing. Run scripts\prepare_candle.ps1 first." }
if ($Device -eq "cuda") {
    $localCuda = Join-Path $repo "tools\cuda-13.0"
    $cudaRoots = @(
        $localCuda,
        $env:CUDA_PATH,
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
    ) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin")) }
    if (-not $cudaRoots) { throw "CUDA runtime was not found." }
    $env:CUDA_PATH = @($cudaRoots)[0]
    $env:CUDA_HOME = $env:CUDA_PATH
    $env:CUDA_ROOT = $env:CUDA_PATH
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    $cudaRuntime = Join-Path $cudaBin "x64"
    $systemCudaRuntime = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
    # The binary links to CUDA 13.4 cuBLASLt but runs kernels compiled with
    # CUDA 13.0, which is compatible with the installed driver.
    if (Test-Path $systemCudaRuntime) {
        $env:Path = "$systemCudaRuntime;$cudaRuntime;$cudaBin;$env:Path"
    } else {
        $env:Path = "$cudaRuntime;$cudaBin;$env:Path"
    }
}
& $binary --model-dir $model --host 127.0.0.1 --port $Port --device $Device

