# Builds the one-click Windows release of Parakeet Dictation.
#
# Produces dist\ParakeetDictation\ (a onedir PyInstaller bundle) with the
# autostart scripts and a plain-English README-FIRST.txt alongside it, then
# zips the whole folder to dist\ParakeetDictation-win64.zip for GitHub Releases.
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\build-exe.ps1
# Needs pyinstaller in the venv:  .venv\Scripts\python.exe -m pip install pyinstaller
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Start clean so stale collected DLLs can't mask a real missing dependency.
if (Test-Path "$PSScriptRoot\build") { Remove-Item "$PSScriptRoot\build" -Recurse -Force }
if (Test-Path "$PSScriptRoot\dist")  { Remove-Item "$PSScriptRoot\dist"  -Recurse -Force }

Write-Host "Running PyInstaller..." -ForegroundColor Cyan

# --windowed: no console (this is a background tray-style app).
# --collect-all pulls in data files, dynamic imports AND bundled DLLs for the
# packages PyInstaller's static analysis can't fully see:
#   onnx_asr        loads its recognizer classes by name at runtime
#   onnxruntime     ships native DLLs (providers) discovered at load time
#   huggingface_hub downloads the model on first run; has lazy submodules
#   sounddevice     wraps a bundled PortAudio DLL under _sounddevice_data
& $py -m PyInstaller `
    --name ParakeetDictation `
    --windowed `
    --noconfirm `
    --collect-all onnx_asr `
    --collect-all onnxruntime `
    --collect-all huggingface_hub `
    --collect-all sounddevice `
    dictation.py
if ($LASTEXITCODE -ne 0) { Write-Host "PyInstaller failed." -ForegroundColor Red; exit 1 }

$out = Join-Path $PSScriptRoot "dist\ParakeetDictation"
if (-not (Test-Path (Join-Path $out "ParakeetDictation.exe"))) {
    Write-Host "Build finished but ParakeetDictation.exe is missing." -ForegroundColor Red
    exit 1
}

Write-Host "Copying autostart scripts and README-FIRST.txt into the bundle..." -ForegroundColor Cyan
Copy-Item (Join-Path $PSScriptRoot "install-autostart.ps1")   $out -Force
Copy-Item (Join-Path $PSScriptRoot "uninstall-autostart.ps1") $out -Force

$readme = @"
Parakeet Dictation - free, unlimited local voice dictation for Windows.

GET STARTED
  1. Double-click ParakeetDictation.exe.
  2. Wait for the first run to download the speech model (about 460 MB, one
     time only). A small "Downloading speech model" pill shows while it works;
     it disappears when the app is ready.
  3. Put your cursor in any text box, then use the hotkeys below.

HOTKEYS
  Ctrl + Win        start listening; press again to stop and paste the text
  Esc               throw away the current recording (types nothing)
  Ctrl + Alt + Q    quit

START IT AUTOMATICALLY AT LOGON (optional)
  Right-click install-autostart.ps1 and pick "Run with PowerShell", then
  approve the one admin prompt. Remove it later with uninstall-autostart.ps1.

WINDOWS SMARTSCREEN
  This exe is not code-signed, so Windows may show a blue "Windows protected
  your PC" screen. Click "More info" then "Run anyway". If you would rather not
  run an unsigned exe from a stranger, the from-source route on the project page
  lets you build and run it yourself with Python instead.

Everything runs on your machine. Your audio never leaves the computer; the only
network use is that one model download on first run.
"@
Set-Content -Path (Join-Path $out "README-FIRST.txt") -Value $readme -Encoding utf8

Write-Host "Zipping..." -ForegroundColor Cyan
$zip = Join-Path $PSScriptRoot "dist\ParakeetDictation-win64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal

$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ""
Write-Host "Done. $zip ($sizeMB MB)" -ForegroundColor Green
