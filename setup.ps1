# One-time setup for Parakeet Dictation (Windows).
# Creates a Python virtual environment next to this script and installs dependencies.
# Run from PowerShell:  .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Find a suitable Python (3.10 - 3.12 recommended; 3.13+ may lack prebuilt wheels
# for some dependencies).
$python = $null
foreach ($candidate in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    try {
        $v = Invoke-Expression "$candidate --version" 2>$null
        if ($v -match "Python 3\.(1[0-2])\.") { $python = $candidate; break }
    } catch {}
}
if (-not $python) {
    Write-Host "No Python 3.10-3.12 found. Install Python 3.12 from https://www.python.org/downloads/ (check 'Add to PATH'), then re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Using: $python ($(Invoke-Expression "$python --version"))" -ForegroundColor Cyan

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
Invoke-Expression "$python -m venv .venv"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet

Write-Host "Installing packages (a few minutes)..." -ForegroundColor Cyan
& $venvPy -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next: run  '.\selftest.py check'  via  .venv\Scripts\python.exe selftest.py"
Write-Host "(first run downloads the speech model, ~460 MB int8, then it is cached)"
Write-Host "Daily use: double-click 'Start Dictation.bat'. See README.md."
