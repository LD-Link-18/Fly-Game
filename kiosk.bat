@echo off
rem Windows stant başlatıcı (kiosk.sh'in karşılığı): çift tıklayınca çalışır.
rem İlk çalıştırmada .venv'i kurar, paketleri yükler ve FlyWire verisini indirir.
rem Oyun çökerse 3 sn sonra yeniden başlatır; operatör ESC'yi basılı tutarak
rem çıkarsa (çıkış kodu 0) döngü biter.
rem
rem Ortam değişkenleriyle değiştirilebilir:
rem   OPPONENTS=flybrain,heuristic   (virgülle birden fazla: ziyaretçi seçer)
rem   LANG_UI=tr                     (arayüz dili; oyunda F3 ile de değişir)
rem   CONFIGS=configs/brain_tuned.toml configs/stand.toml
rem   TORCH_INDEX=https://download.pytorch.org/whl/cu128   (PyTorch CUDA sürümü)
rem Örnek (cmd penceresinde): önce "set LANG_UI=tr", sonra "kiosk.bat"
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not defined OPPONENTS set "OPPONENTS=flybrain,heuristic"
if not defined LANG_UI set "LANG_UI=en"
if not defined CONFIGS set "CONFIGS=configs/brain_tuned.toml configs/stand.toml"
if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "VPY=.venv\Scripts\python.exe"

rem Kurulum tamamlanınca işaret dosyası yazılır; yarım kalan kurulum tekrar denenir.
if exist ".venv\.setup-ok" goto run

echo First run: setting up (Python packages ~3 GB, FlyWire data ~137 MB^)...
if exist "%VPY%" goto install
rem "py" başlatıcısını tercih et; Microsoft Store'un sahte "python"unu sürüm kontrolü eler.
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.11+ not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then run kiosk.bat again.
  pause
  exit /b 1
)
%PY% -m venv .venv || goto setup_failed

:install
"%VPY%" -m pip install --upgrade pip || goto setup_failed
"%VPY%" -m pip install -r requirements.txt || goto setup_failed
rem PyPI'deki Windows torch'u yalnızca CPU'dur; CUDA sürümü PyTorch'un kendi dizininden gelir.
"%VPY%" -m pip install torch --index-url %TORCH_INDEX% || goto setup_failed
"%VPY%" -m pip install -r requirements-brain.txt || goto setup_failed
"%VPY%" -m flygame brain-download || goto setup_failed
echo ok> ".venv\.setup-ok"

:run
set "ARGS="
for %%c in (%CONFIGS%) do set "ARGS=!ARGS! --config %%c"

:loop
"%VPY%" -m flygame play --kiosk --opponent "%OPPONENTS%" --lang %LANG_UI% %ARGS% %*
set "code=%errorlevel%"
if "%code%"=="0" exit /b 0
echo Game exited with code %code%; restarting in 3 s (Ctrl+C to stop^) 1>&2
timeout /t 3 /nobreak >nul
goto loop

:setup_failed
echo.
echo Setup failed (see the messages above^). Check the internet connection and run kiosk.bat again.
pause
exit /b 1
