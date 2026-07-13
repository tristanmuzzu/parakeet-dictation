# Removes the auto-start task registered by install-autostart.ps1.
# Self-elevates if not run as administrator.
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Elevation required - approve the UAC prompt..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","`"$PSCommandPath`""
    exit
}

try { Stop-ScheduledTask -TaskName "ParakeetDictation" -ErrorAction SilentlyContinue } catch {}
Unregister-ScheduledTask -TaskName "ParakeetDictation" -Confirm:$false
Write-Host "Auto-start removed. (The app itself and your model cache are untouched.)" -ForegroundColor Green
pause
