@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v9
if errorlevel 1 exit /b 1
echo ZIP hasil ada di campus-results.
