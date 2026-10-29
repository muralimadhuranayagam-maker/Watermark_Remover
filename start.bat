@echo off
setlocal

echo ======================================================
echo  Launching Video Watermark Remover
echo ======================================================

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Running setup.bat first...
    call setup.bat
    if errorlevel 1 exit /b 1
)

:: Ensure temp dir exists
if not exist "C:\wmtool\temp" mkdir "C:\wmtool\temp" 2>nul
if not exist "models" mkdir "models" 2>nul

:: Check FFmpeg in winget packages if not in PATH
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do (
        if exist "%%D\ffmpeg-*\bin\ffmpeg.exe" (
            for /d %%B in ("%%D\ffmpeg-*\bin") do set "PATH=%%B;%PATH%"
        )
    )
)

call .\venv\Scripts\activate.bat
python app.py
if errorlevel 1 (
    echo [ERROR] Application exited with an error.
    pause
)
