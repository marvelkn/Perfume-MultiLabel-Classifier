@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v7
exit /b %errorlevel%
