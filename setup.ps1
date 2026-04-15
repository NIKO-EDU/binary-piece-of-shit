# setup.ps1 — Windows/PowerShell setup for SOD-OPT
#
# Usage (from the repo root in PowerShell):
#   .\setup.ps1            # auto-detect CUDA
#   .\setup.ps1 cpu        # force CPU-only PyTorch
#   .\setup.ps1 cu121      # CUDA 12.x
#   .\setup.ps1 cu118      # CUDA 11.8
#
# Prerequisites:
#   1. Python 3.10+   — https://www.python.org/downloads/
#   2. gcc (MinGW-w64) — needed to compile the C kernel:
#        winget install MSYS2.MSYS2
#      Then from the MSYS2 MinGW64 shell run:
#        pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp make
#      Then add C:\msys64\mingw64\bin to your PATH.
#
# If PowerShell blocks execution, run once:
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

param(
    [string]$CudaBuild = "auto"
)

$ErrorActionPreference = "Stop"

# ── Helper ────────────────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host "[setup] $msg" -ForegroundColor Cyan
}

function Write-Warn([string]$msg) {
    Write-Host "[setup] WARNING: $msg" -ForegroundColor Yellow
}

# ── 0. Execution-policy check ────────────────────────────────────────────────
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Warn "PowerShell execution policy is Restricted."
    Write-Warn "Run this once to fix it:"
    Write-Warn "  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}

# ── 1. Find Python ────────────────────────────────────────────────────────────
$pythonCmd = $null
foreach ($cmd in @("python", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $pythonCmd = $cmd
        break
    }
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
$pyVersion = & $pythonCmd --version 2>&1
Write-Step "Using $pyVersion ($pythonCmd)"

# ── 2. Create venv ────────────────────────────────────────────────────────────
if (-not (Test-Path "venv")) {
    Write-Step "Creating virtual environment..."
    & $pythonCmd -m venv venv
} else {
    Write-Step "venv already exists, skipping creation."
}

# Activate
$activateScript = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: venv activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}
. $activateScript
Write-Step "venv activated."
python -m pip install --quiet --upgrade pip

# ── 3. Detect CUDA ────────────────────────────────────────────────────────────
if ($CudaBuild -eq "auto") {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    $nsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

    if ($nvcc) {
        $nvccOut = & nvcc --version 2>&1 | Out-String
        if ($nvccOut -match "release (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            if ($major -ge 12) {
                $CudaBuild = "cu121"
            } elseif ($major -eq 11) {
                $CudaBuild = "cu118"
            } else {
                $CudaBuild = "cpu"
            }
            Write-Step "Detected CUDA $($Matches[1]).$($Matches[2]) -> using $CudaBuild build"
        } else {
            $CudaBuild = "cu121"
            Write-Step "nvcc found but version unreadable, assuming cu121"
        }
    } elseif ($nsmi) {
        $CudaBuild = "cu121"
        Write-Step "nvidia-smi found, assuming CUDA 12.x -> using cu121 build"
    } else {
        $CudaBuild = "cpu"
        Write-Step "No CUDA detected -> using CPU-only PyTorch"
    }
}

# ── 4. Install PyTorch ────────────────────────────────────────────────────────
if ($CudaBuild -eq "cpu") {
    Write-Step "Installing CPU-only PyTorch..."
    pip install torch torchvision
} else {
    Write-Step "Installing PyTorch with $CudaBuild..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaBuild"
}

# ── 5. Install remaining dependencies ────────────────────────────────────────
Write-Step "Installing remaining dependencies from requirements.txt..."
$deps = Get-Content requirements.txt |
    Where-Object { $_ -notmatch '^\s*#' -and $_ -notmatch '^\s*$' -and $_ -notmatch '^torch' }
if ($deps) {
    $deps | pip install --quiet -r /dev/stdin 2>$null
    # Fallback for systems where /dev/stdin doesn't exist
    if ($LASTEXITCODE -ne 0) {
        $tmpFile = [System.IO.Path]::GetTempFileName()
        $deps | Set-Content $tmpFile
        pip install --quiet -r $tmpFile
        Remove-Item $tmpFile
    }
}

# ── 6. Build C kernel ─────────────────────────────────────────────────────────
Write-Step "Building C kernel (make)..."
$gccAvailable = Get-Command gcc -ErrorAction SilentlyContinue
$makeAvailable = Get-Command make -ErrorAction SilentlyContinue

if ($gccAvailable -and $makeAvailable) {
    make
    if ($LASTEXITCODE -eq 0) {
        Write-Step "kernel.dll built successfully."
    } else {
        Write-Warn "make failed. Check gcc/OpenMP installation."
    }
} else {
    Write-Warn "gcc or make not found. The C kernel will not be available."
    Write-Host ""
    Write-Host "  To install MinGW-w64 (gcc for Windows):" -ForegroundColor Yellow
    Write-Host "    winget install MSYS2.MSYS2" -ForegroundColor Yellow
    Write-Host "  Then open the MSYS2 MinGW64 shell and run:" -ForegroundColor Yellow
    Write-Host "    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp make" -ForegroundColor Yellow
    Write-Host "  Then add C:\msys64\mingw64\bin to your PATH and re-run setup.ps1." -ForegroundColor Yellow
}

# ── 7. Summary ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  # Download dataset (if not already present):"
Write-Host "  #   https://github.com/VisDrone/VisDrone-Dataset"
Write-Host "  #   -> extract to data\VisDrone2019-DET-train\"
Write-Host ""
Write-Host "  # (Optional) Download YOLOv8 weights for benchmarking:"
Write-Host "  Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt' -OutFile yolov8n.pt"
Write-Host "  Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx' -OutFile yolov8n.onnx"
Write-Host ""
Write-Host "  # Train:"
Write-Host "  python scripts\train_bnn.py --device cuda --epochs 20"
Write-Host "  python scripts\train_bnn.py --device cuda --epochs 20 --max-samples 100  # smoke-test"
