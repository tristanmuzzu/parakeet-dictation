# Update Parakeet Dictation and actually put the new code live.
#
# Right-click -> "Run with PowerShell", or:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\update.ps1
#
# Pulling new code is not enough on its own and that trips everyone up: the app
# auto-starts at logon and keeps the OLD code in memory until it is restarted,
# and a packaged ParakeetDictation.exe ignores source changes entirely because
# it bundles its own Python. This script does the whole job and, more
# importantly, TELLS YOU which of those situations you are in instead of
# silently doing nothing.

$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$taskName = "ParakeetDictation"
$problems = @()

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Head($msg) { Write-Host ""; Write-Host $msg -ForegroundColor Cyan }

Head "Parakeet Dictation - update"
Say "Install folder: $dir"

# --- 1. Where is the copy that actually RUNS? -------------------------------
# If auto-start points at a different folder than this one, everything else in
# this script is beside the point, so check it before touching anything.
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$taskDir = $null
if ($task) {
    $action = $task.Actions | Select-Object -First 1
    $taskDir = $action.WorkingDirectory
    if (-not $taskDir -and $action.Execute) {
        $taskDir = Split-Path -Parent $action.Execute
    }
    Say "Auto-start task: registered, runs from $taskDir"
    if ($taskDir -and (Resolve-Path $taskDir -ErrorAction SilentlyContinue).Path -ne (Resolve-Path $dir).Path) {
        Say ""
        Say "  *** The app that starts with Windows lives in a DIFFERENT folder" -ForegroundColor Red
        Say "  ***   runs from : $taskDir" -ForegroundColor Red
        Say "  ***   you are in: $dir" -ForegroundColor Red
        Say "  *** Updating here will never change what you actually use." -ForegroundColor Red
        Say "  *** Either run this script from that folder, or re-register" -ForegroundColor Red
        Say "  *** auto-start here with install-autostart.ps1." -ForegroundColor Red
        $problems += "auto-start points at $taskDir"
    }
} else {
    Say "Auto-start task: not registered (you launch it by hand)"
}

# --- 2. Packaged exe? Source updates cannot reach it. -----------------------
$exe = Join-Path $dir "ParakeetDictation.exe"
$hasExe = Test-Path $exe
if ($hasExe) {
    Say ""
    Say "  ParakeetDictation.exe is present in this folder." -ForegroundColor Yellow
    Say "  The exe has Python baked inside it, so pulling source does NOT" -ForegroundColor Yellow
    Say "  update it. Rebuild with build-exe.ps1, or download the newest" -ForegroundColor Yellow
    Say "  ParakeetDictation-win64.zip from the GitHub Releases page." -ForegroundColor Yellow
    $problems += "packaged exe present - rebuild or re-download it"
}

# --- 3. Pull the newest code ------------------------------------------------
Head "Fetching latest code"
if (Test-Path (Join-Path $dir ".git")) {
    Push-Location $dir
    try {
        $branch = (git rev-parse --abbrev-ref HEAD).Trim()
        Say "Current branch: $branch"
        if ($branch -ne "main") {
            Say "Switching to main..."
            git checkout main
        }
        git pull --ff-only origin main
        $head = (git log --oneline -1).Trim()
        Say "Now at: $head" "Green"
    } catch {
        Say "git pull failed: $_" "Red"
        $problems += "git pull failed"
    } finally {
        Pop-Location
    }
} else {
    Say "Not a git checkout - nothing to pull here." "Yellow"
    if (-not $hasExe) { $problems += "no git checkout and no exe in this folder" }
}

# --- 4. Prove the new code is present, without needing a mic ----------------
Head "Checking the text pipeline"
$py = Join-Path $dir ".venv\Scripts\python.exe"
if (Test-Path $py) {
    & $py (Join-Path $dir "tools\formatcheck.py") --quiet
    if ($LASTEXITCODE -ne 0) { $problems += "formatcheck reported the feature is missing" }
} else {
    Say "No .venv here (exe install, or setup.ps1 was never run)." "Yellow"
}

# --- 5. Restart, so the running process picks the new code up ---------------
Head "Restarting the app"
if ($task) {
    try {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Start-ScheduledTask -TaskName $taskName
        Say "Scheduled task restarted." "Green"
    } catch {
        Say "Could not restart the task: $_" "Red"
        $problems += "task restart failed"
    }
} else {
    Get-Process pythonw, python, ParakeetDictation -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($dir) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    if ($hasExe) {
        Start-Process -FilePath $exe -WorkingDirectory $dir
        Say "Relaunched ParakeetDictation.exe." "Green"
    } elseif (Test-Path (Join-Path $dir ".venv\Scripts\pythonw.exe")) {
        Start-Process -FilePath (Join-Path $dir ".venv\Scripts\pythonw.exe") `
                      -ArgumentList "dictation.py" -WorkingDirectory $dir
        Say "Relaunched from source." "Green"
    } else {
        Say "Nothing to launch here." "Red"
        $problems += "no exe and no venv to launch"
    }
}

# --- 6. Confirm from the log what is actually live --------------------------
Head "Verifying"
Start-Sleep -Seconds 3
$logPath = Join-Path $dir "dictation.log"
if (Test-Path $logPath) {
    $line = Get-Content $logPath -Tail 80 |
            Where-Object { $_ -match "text pipeline:" } |
            Select-Object -Last 1
    if ($line) {
        Say $line.Trim() "Green"
    } else {
        Say "No 'text pipeline:' line in the log yet." "Yellow"
        Say "Give it a few seconds and check dictation.log again. If it never" "Yellow"
        Say "appears, the running copy is still the old build." "Yellow"
        $problems += "no pipeline line in the log"
    }
} else {
    Say "No dictation.log in this folder - the app may run from elsewhere." "Yellow"
}

Head "Result"
if ($problems.Count -eq 0) {
    Say "Up to date and running. Press Ctrl+Win and try it." "Green"
} else {
    Say "Finished, but these need your attention:" "Red"
    foreach ($p in $problems) { Say "  - $p" "Red" }
}
Write-Host ""
Read-Host "Press Enter to close"
