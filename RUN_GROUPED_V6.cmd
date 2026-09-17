@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ikuti instalasi sekali di WINDOWS_GUIDE_V6.md.
  exit /b 1
)
".venv\Scripts\python.exe" -u -B -m alignment.experiments_v6
if errorlevel 1 (
  echo Proses belum selesai. Simpan folder ini; jalankan perintah yang sama untuk resume.
  exit /b 1
)
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v6
if errorlevel 1 exit /b 1
echo SELESAI. Bawa pulang ZIP hasil terbaru dari folder campus-results.
