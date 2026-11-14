param(
    [string]$Adapter = "models\wisp-v0.1.2-4b",
    [string]$Python = "python",
    [switch]$ForceMerge,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = $Python
$adapter = if ([System.IO.Path]::IsPathRooted($Adapter)) { $Adapter } else { Join-Path $repo $Adapter }
$model = Join-Path $repo "models\wisp-v0.1.2-4b-candle"

if (-not (Get-Command $python -ErrorAction SilentlyContinue)) { throw "Python not found: $python" }
if (-not (Test-Path (Join-Path $adapter "adapter_model.safetensors"))) { throw "Wisp adapter not found: $adapter (download it with: hf download notebox/Wisp-v0.1.2-4B --local-dir models\wisp-v0.1.2-4b)" }

$mergeArgs = @("$repo\scripts\merge_adapter_for_candle.py", "--adapter", $adapter, "--output", $model)
if ($ForceMerge) { $mergeArgs += "--force" }
& $python @mergeArgs
if ($LASTEXITCODE -ne 0) { throw "Adapter merge failed with exit code $LASTEXITCODE" }

$features = if ($CpuOnly) { "inference" } else { "cuda" }
if (-not $CpuOnly) {
    $localCuda = Join-Path $repo "tools\cuda-13.0"
    $cudaRoots = @(
        $localCuda,
        $env:CUDA_PATH,
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
    ) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin\nvcc.exe")) }
    if (-not $cudaRoots) { throw "CUDA Toolkit with nvcc is required for the native GPU build." }
    $env:CUDA_PATH = @($cudaRoots)[0]
    $env:CUDA_HOME = $env:CUDA_PATH
    $env:CUDA_ROOT = $env:CUDA_PATH
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    $cudaRuntime = Join-Path $cudaBin "x64"
    $env:Path = "$cudaRuntime;$cudaBin;$env:Path"
    if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
        throw "nvcc is not visible after configuring CUDA_PATH=$env:CUDA_PATH"
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools locator was not found." }
        $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
        if (-not (Test-Path $vsDevCmd)) { throw "VsDevCmd.bat was not found in $vsInstall" }
        $environment = & cmd.exe /s /c "`"$vsDevCmd`" -no_logo -arch=x64 && set"
        foreach ($line in $environment) {
            if ($line -match '^([^=]+)=(.*)$') {
                Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
            }
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC cl.exe is required by nvcc and is still not visible."
    }
    # CUDA 13's CCCL headers reject MSVC's legacy preprocessor. CL is honored
    # by the host compiler invoked through nvcc.
    $env:CL = "/Zc:preprocessor"
    $env:CUDA_COMPUTE_CAP = "89"
    # Candle's current CUDA binding (cudarc 0.19) references cuBLASLt
    # emulation symbols introduced after CUDA 13.0. The RTX 4070 driver on
    # this machine accepts CUDA 13.0 PTX, so compile kernels with our local
    # CUDA 13.0 toolchain, but use the already-installed 13.4 cuBLAS import
    # libraries solely for the host link step. This does not change drivers
    # or model weights.
    if ($env:CUDA_PATH -eq $localCuda) {
        $systemCublas = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
        $required = @("cublas.lib", "cublasLt.lib")
        if (-not (Test-Path (Join-Path $systemCublas "lib\x64\cublasLt.lib"))) {
            throw "Candle/cudarc requires newer cuBLASLt symbols, but CUDA 13.4 cuBLAS is unavailable."
        }
        $backup = Join-Path $localCuda "compat-backup-cublas13.0"
        New-Item -ItemType Directory -Force -Path $backup | Out-Null
        foreach ($name in $required) {
            $target = Join-Path $localCuda "lib\x64\$name"
            $saved = Join-Path $backup $name
            if (-not (Test-Path $saved)) { Copy-Item -LiteralPath $target -Destination $saved }
            Copy-Item -LiteralPath (Join-Path $systemCublas "lib\x64\$name") -Destination $target -Force
        }
        $systemRuntime = Join-Path $systemCublas "bin\x64"
        $env:Path = "$systemRuntime;$env:Path"
    }
    Write-Host "CUDA compiler: $((Get-Command nvcc.exe).Source)"
    Write-Host "MSVC compiler: $((Get-Command cl.exe).Source)"
}

& cargo build --release --manifest-path "$repo\rust\Cargo.toml" --features $features --bin wisp-server
if ($LASTEXITCODE -ne 0) { throw "Rust/Candle build failed with exit code $LASTEXITCODE" }
Write-Host "Prepared the Wisp v0.1.2 4B Candle runtime. Model: $model"

