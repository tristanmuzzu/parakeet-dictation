# Registers Parakeet Dictation to start automatically at logon (Task Scheduler).
# Self-elevates if not run as administrator.
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Elevation required - approve the UAC prompt..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","`"$PSCommandPath`""
    exit
}

$dir = $PSScriptRoot
$pyw = Join-Path $dir ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) {
    Write-Host "Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    pause; exit 1
}
$user = "$env:USERDOMAIN\$env:USERNAME"

$action    = New-ScheduledTaskAction -Execute $pyw -Argument "dictation.py" -WorkingDirectory $dir
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName "ParakeetDictation" -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered 'ParakeetDictation' to auto-start at logon." -ForegroundColor Green
Start-ScheduledTask -TaskName "ParakeetDictation"
Write-Host "Started it now. Give it ~15-30 seconds to load the model, then press Ctrl+Win."
pause
