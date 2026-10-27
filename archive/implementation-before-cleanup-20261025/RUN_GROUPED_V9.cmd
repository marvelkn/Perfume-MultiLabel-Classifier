@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=6
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Menyiapkan Python 3.12. Internet diperlukan saat instalasi pertama.
  py -3.12 -m venv .venv
  if errorlevel 1 goto setup_error
)
if not exist ".venv\v9_dependencies_ready.txt" (
  ".venv\Scripts\python.exe" -m pip install -r requirements_campus_v9.txt
  if errorlevel 1 goto setup_error
  ".venv\Scripts\python.exe" -u -B -m alignment.experiments_v9 --check
  if errorlevel 1 goto run_error
  echo V9 dependencies verified>".venv\v9_dependencies_ready.txt"
)
echo V9: 8 kandidat. Tuning 64 jam total, ditambah validasi dan fitting final.
echo CPU 6 thread, model berurutan. Untuk resume, jalankan file ini lagi.
".venv\Scripts\python.exe" -u -B -m alignment.experiments_v9
if errorlevel 1 goto run_error
".venv\Scripts\python.exe" -u -B -m alignment.collect_results_v9
if errorlevel 1 goto run_error
echo SELESAI. Bawa ZIP terbaru dari folder campus-results.
pause
exit /b 0
:setup_error
echo Instalasi belum berhasil. Pastikan Python 3.12 64-bit tersedia dan internet aktif.
echo Baca WINDOWS_GUIDE_V9.md, lalu jalankan file ini lagi.
pause
exit /b 1
:run_error
echo Proses belum selesai. Simpan pesan error dan folder ini.
echo Untuk resume pada PC yang sama, jalankan RUN_GROUPED_V9.cmd lagi.
pause
exit /b 1
