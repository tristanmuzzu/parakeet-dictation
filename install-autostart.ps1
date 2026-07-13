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
$user = "$env:USERDOMAIN\$env:USERNAME"

# Prefer the bundled exe (one-click build) if it sits next to this script;
# otherwise fall back to the from-source venv. The two routes share one model
# cache and the single-instance lock, so whichever we register, only one copy
# ever runs.
$exe = Join-Path $dir "ParakeetDictation.exe"
if (Test-Path $exe) {
    Write-Host "Found ParakeetDictation.exe - registering the packaged build." -ForegroundColor Cyan
    $action = New-ScheduledTaskAction -Execute $exe -WorkingDirectory $dir
} else {
    $pyw = Join-Path $dir ".venv\Scripts\pythonw.exe"
    if (-not (Test-Path $pyw)) {
        Write-Host "No ParakeetDictation.exe and no virtual environment. Run setup.ps1 first, or use the packaged build." -ForegroundColor Red
        pause; exit 1
    }
    Write-Host "Registering the from-source build (.venv)." -ForegroundColor Cyan
    $action = New-ScheduledTaskAction -Execute $pyw -Argument "dictation.py" -WorkingDirectory $dir
}
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
