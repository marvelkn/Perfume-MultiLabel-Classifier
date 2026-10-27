@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v6
exit /b %errorlevel%
