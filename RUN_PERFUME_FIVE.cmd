@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Buat .venv dan instal requirements_training.txt sesuai WINDOWS_GUIDE.md.
  exit /b 1
)
if not exist "data\builds\perfume-five-grouped-v2\dataset_manifest.json" (
  ".venv\Scripts\python.exe" -B -m alignment.perfume_data
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -B -m alignment.experiments --check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -B -m alignment.experiments
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -B -m alignment.collect_results
if errorlevel 1 exit /b 1
echo SELESAI. Bawa folder runs\perfume-five-grouped-v2 dan data\builds\perfume-five-grouped-v2.
