@echo off
REM ============================================================
REM  Launch QVista.
REM
REM  Double-click this file, or run it from a terminal.
REM
REM  It changes to its own folder first, so QVista is found
REM  whether or not the package has been installed into the
REM  environment.
REM
REM  python.exe is used rather than pythonw.exe so this console
REM  window stays open and shows any error. Close the window to
REM  quit QVista, or close QVista itself.
REM ============================================================

cd /d "%~dp0"

REM .venv is what the README tells you to create. .venv-win is
REM kept for environments made before that was the documented
REM name, so an existing checkout kept working.
set "PYTHON="

if exist ".venv\Scripts\python.exe"     set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON (
    if exist ".venv-win\Scripts\python.exe" set "PYTHON=.venv-win\Scripts\python.exe"
)

if not defined PYTHON (
    echo.
    echo   ERROR: no virtual environment found.
    echo   Looked for: %~dp0.venv\Scripts\python.exe
    echo               %~dp0.venv-win\Scripts\python.exe
    echo.
    echo   Create one with:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -e .
    echo.
    echo   See the Installation section of README.md for the full steps.
    echo.
    pause
    exit /b 1
)

echo Starting QVista...
echo.

"%PYTHON%" -m qvista %*

if errorlevel 1 (
    echo.
    echo   QVista exited with an error. The message above says why.
    echo.
    pause
)
