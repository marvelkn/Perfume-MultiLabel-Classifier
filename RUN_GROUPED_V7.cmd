@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ikuti instalasi sekali di WINDOWS_GUIDE_V7.md.
  exit /b 1
)
".venv\Scripts\python.exe" -u -B -m alignment.experiments_v7
if errorlevel 1 (
  echo Proses belum selesai. Simpan folder ini dan jalankan lagi untuk resume.
  exit /b 1
)
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v7
if errorlevel 1 exit /b 1
echo SELESAI. Bawa ZIP terbaru dari folder campus-results.
