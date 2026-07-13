@echo off
REM First-run / troubleshooting launcher: keeps a console window open so you
REM can see the model download progress and any errors.
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" dictation.py
echo.
echo (Dictation stopped. Press any key to close.)
pause >nul
