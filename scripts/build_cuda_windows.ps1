# Build ik_llama.cpp + Gemma 4 MTP patches with CUDA on Windows.
#
# Pre-req:
#   - Visual Studio 2022 Build Tools (MSVC 14.3x) with "Desktop development with C++"
#   - CUDA Toolkit 12.6 (12.4+ works, but 12.6 has the best Ada/Ampere kernels)
#   - cmake >= 3.20 (Visual Studio bundles 3.30 in BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin)
#   - git
#
# Tip: launch from "x64 Native Tools Command Prompt for VS 2022" so cl.exe and link.exe are on PATH.
# Tip: pwsh.exe (PowerShell 7) is preferred but Windows PowerShell 5.1 also works.
#
# Usage (from repo root):
#   pwsh -ExecutionPolicy Bypass -File scripts\build_cuda_windows.ps1
#   pwsh ... -CudaArch 89             # Ada (4090)
#   pwsh ... -CudaArch "86;89"        # multi-arch fat binary
#   pwsh ... -Source D:\src\ik_llama.cpp -Jobs 16

param(
  [string]$Source   = "$(Resolve-Path "$PSScriptRoot\..").Path\build\ik_llama.cpp",
  [string]$CudaArch = "86",   # default sm_86 covers RTX 3090 / 3090 Ti
  [int]   $Jobs     = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"

function Log { param([string]$msg) Write-Host "[build_cuda_windows] $msg" -ForegroundColor Cyan }
function Err { param([string]$msg) Write-Host "[build_cuda_windows] ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Source)) { Err "Source tree not found at $Source. Run scripts/apply_patches.sh in WSL or a Git-Bash shell first." }

# Pre-flight
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) { Err "cmake not found. Install via Visual Studio Installer or 'winget install Kitware.CMake'." }

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
  if (Test-Path "$env:CUDA_PATH\bin\nvcc.exe") {
    $env:Path = "$env:CUDA_PATH\bin;" + $env:Path
    Log "added $env:CUDA_PATH\bin to PATH"
  } else {
    Err "nvcc not found and `$env:CUDA_PATH not set. Install CUDA Toolkit 12.x."
  }
}

Log "cmake:  $((cmake --version | Select-Object -First 1))"
Log "nvcc:   $((nvcc --version | Select-String release | Select-Object -First 1))"
Log "src:    $Source"
Log "arch:   $CudaArch"
Log "jobs:   $Jobs"

Push-Location $Source
try {
  if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
  Set-Location "build"

  Log "configuring (Release, CUDA, server, full kernels)"
  # Note: -DCMAKE_GENERATOR_TOOLSET="cuda=$env:CUDA_PATH" forces VS to use the matching CUDA toolset.
  $cfgArgs = @(
    "..",
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-T", "host=x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DGGML_CUDA_F16=ON",
    "-DGGML_NATIVE=ON",
    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
  )
  & cmake @cfgArgs
  if ($LASTEXITCODE -ne 0) { Err "cmake configure failed" }

  Log "compiling (parallel x$Jobs)"
  & cmake --build . --config Release --parallel $Jobs --target llama-server llama-cli llama-bench
  if ($LASTEXITCODE -ne 0) { Err "build failed" }

  $binDir = "bin\Release"
  if (-not (Test-Path "$binDir\llama-server.exe")) {
    # Some VS configs put binaries directly in bin\
    $binDir = "bin"
  }
  Log "build complete - binaries in $((Get-Location).Path)\$binDir"
  Get-ChildItem "$binDir\llama-server.exe","$binDir\llama-cli.exe","$binDir\llama-bench.exe" -ErrorAction SilentlyContinue |
    ForEach-Object { Log ("  " + $_.FullName) }

  Log "smoke test:"
  & ".\$binDir\llama-server.exe" --version | Select-Object -First 3 | ForEach-Object { Log "  $_" }
}
finally {
  Pop-Location
}

Log "done. Try (single line):"
Log "  llama-server.exe -m gemma-4-31B-Q8_0.gguf --spec-type mtp -md gemma-4-31B-it-assistant-Q8_0.gguf -ngl 99 -ngld 99 --draft-max 3 --port 8005"
