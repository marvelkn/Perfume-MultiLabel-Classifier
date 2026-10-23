@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
echo V8 - Audit dataset Suh dan fitur Morgan 2048 + 8. Tidak training.
if exist ".venv-v8\Scripts\python.exe" goto checkenv
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 goto nopython
py -3.12 -m venv .venv-v8
if errorlevel 1 goto failed
:checkenv
".venv-v8\Scripts\python.exe" -B -c "from v8.prepare import validate_environment; validate_environment()" >nul 2>&1
if not errorlevel 1 goto run
".venv-v8\Scripts\python.exe" -m pip install -r requirements_preml_v8.txt
if errorlevel 1 goto failed
:run
".venv-v8\Scripts\python.exe" -B -m v8.prepare run
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="2" goto review
if not "%RESULT%"=="0" goto failed
goto review
:review
echo.
echo AUDIT SELESAI. STOP SEBELUM ML - perlu review temuan.
echo Buka data\builds\suh-ten-v8-m2048-d8-preml\HASIL_AUDIT_V8.md
echo Tidak ada training/tuning. Simpan seluruh folder hasil audit.
pause
exit /b 2
:nopython
echo Python 3.12 belum tersedia. Install Python 3.12 dari python.org, lalu jalankan lagi.
pause
exit /b 1
:failed
echo.
echo Ada error teknis. Simpan pesan error di atas. Jangan hapus atau timpa hasil.
pause
exit /b 1
