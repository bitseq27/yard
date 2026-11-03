@echo off
setlocal enabledelayedexpansion
title Pixel Sway
cd /d "%~dp0"

rem ===========================================================================
rem  Pixel Sway launcher (Windows)
rem  Keep this file next to the app\ folder. Double-click it, or run
rem     launch.bat [port]     (default port 8000)
rem  Requirements: Python 3.9+ with pip (numpy/Pillow auto-install if missing).
rem  Optional later: gifsicle / ffmpeg on PATH for optimized exports.
rem ===========================================================================

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

rem ---- find python ----------------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3 not found.
  echo         Install it from https://python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Using: %PY%
%PY% --version

rem ---- dependencies ---------------------------------------------------------
%PY% -c "import numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo Installing numpy + Pillow ^(first run only^) ...
  %PY% -m pip install --user numpy pillow
  if errorlevel 1 (
    echo [ERROR] Automatic install failed. Run manually:
    echo         %PY% -m pip install numpy pillow
    pause
    exit /b 1
  )
)

rem ---- optional tools (future milestones) -----------------------------------
where gifsicle >nul 2>nul || echo [note] gifsicle not on PATH - lossless GIF optimization off until installed.
where ffmpeg   >nul 2>nul || echo [note] ffmpeg not on PATH   - MP4 export unavailable ^(not needed yet^).

rem ---- port check -----------------------------------------------------------
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo [ERROR] Port %PORT% is already in use.
  echo         Either stop that program or run:  launch.bat 8001
  pause
  exit /b 1
)

rem ---- launch ----------------------------------------------------------------
echo.
echo   Pixel Sway editor:  http://127.0.0.1:%PORT%/
echo   This window shows server logs. Press Ctrl+C (or close it) to stop.
echo.
start /b "" powershell -NoProfile -Command "Start-Sleep 2; Start-Process 'http://127.0.0.1:%PORT%/'"
%PY% app\server.py %PORT%

echo.
echo Server stopped.
pause
