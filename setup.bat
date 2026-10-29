@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo  Watermark Remover - Setup Script
echo ======================================================

:: 1. Check Python
set "PYTHON_CMD="
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
) else (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
)

if "!PYTHON_CMD!"=="" (
    echo [ERROR] Python 3.11 is not found on PATH or in default install directory.
    echo Please install Python 3.11 or add it to your PATH.
    pause
    exit /b 1
)

echo [OK] Using Python: !PYTHON_CMD!
!PYTHON_CMD! --version

:: 2. Check FFmpeg
ffmpeg -version >nul 2>&1
if !errorlevel! neq 0 (
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
        if exist "%%D\ffmpeg-*\bin\ffmpeg.exe" (
            for /d %%B in ("%%D\ffmpeg-*\bin") do set "PATH=%%B;!PATH!"
        )
    )
)
ffmpeg -version >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARNING] FFmpeg is not found on your current PATH.
    echo If you recently installed FFmpeg via winget, please restart your terminal.
    echo Otherwise, run: winget install Gyan.FFmpeg
) else (
    echo [OK] FFmpeg is available.
)

:: 3. Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating Python virtual environment in .\venv ...
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Existing virtual environment found.
)

:: 4. Install Dependencies
echo [INFO] Upgrading pip and installing dependencies...
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)

:: 5. Create required directories
if not exist "models" mkdir "models"
if not exist "C:\wmtool\temp" mkdir "C:\wmtool\temp" 2>nul

echo ======================================================
echo [SUCCESS] Setup completed successfully!
echo You can now run start.bat to launch Watermark Remover.
echo ======================================================
pause
