@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment belum tersedia. Ikuti WINDOWS_GUIDE.md bagian setup.
  pause
  exit /b 1
)
set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" -u "reports\campus_20260909\full\run_all.py" --campus-no-temperature --run "runs/campus-full-main"
set RUN_RESULT=%ERRORLEVEL%
echo.
echo Exit code: %RUN_RESULT%
echo Hasil ZIP ada di folder campus-results. Lihat RESULTS_SUMMARY.json di dalam ZIP.
echo Jika proses tadi terputus, jalankan file ini lagi untuk resume pada run yang sama.
pause
exit /b %RUN_RESULT%
