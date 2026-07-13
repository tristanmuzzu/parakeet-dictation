@echo off
REM Normal launcher: runs in the background with no console window.
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" dictation.py
