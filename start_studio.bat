@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Set working directory to project root
cd /d "%~dp0"

echo [VoxAI Studio] Checking local environment...

:: 1. Prefer local portable runtime if exists
if exist "runtime\python.exe" (
    echo [System] Found internal portable runtime. Using it.
    set "PYTHON_EXE=%~dp0runtime\python.exe"
    set "VIRTUAL_ENV=%~dp0runtime"
) else (
    if exist "runtime\Scripts\python.exe" (
        echo [System] Found internal portable runtime ^(venv layout^). Using it.
        set "PYTHON_EXE=%~dp0runtime\Scripts\python.exe"
        set "VIRTUAL_ENV=%~dp0runtime"
        set "PATH=%~dp0runtime\Scripts;!PATH!"
    ) else (
        if exist ".venv\Scripts\python.exe" (
            echo [System] Found .venv. Using it.
            set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
            set "VIRTUAL_ENV=%~dp0.venv"
            set "PATH=%~dp0.venv\Scripts;!PATH!"
        ) else (
            echo [System] No local environment found. Relying on system python.
            set "PYTHON_EXE=python"
        )
    )
)

:: 2. Prefer bundled Node.js/npm if exists
if exist "%~dp0tools\node\npm.cmd" (
    echo [System] Found bundled Node.js/npm. Using it.
    set "NODE_HOME=%~dp0tools\node"
    set "PATH=!NODE_HOME!;!PATH!"
    set "NPM_CMD=!NODE_HOME!\npm.cmd"
) else (
    set "NPM_CMD=npm"
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] npm not found. Please install Node.js or add bundled Node.js to "%~dp0tools\node".
        pause
        exit /b 1
    )
)

:: 3. HuggingFace download fallback endpoints/timeouts
if not defined HF_ENDPOINTS (
    set "HF_ENDPOINTS=https://huggingface.co;https://hf-mirror.com"
)
if not defined HF_HUB_ETAG_TIMEOUT (
    set "HF_HUB_ETAG_TIMEOUT=60"
)
if not defined HF_HUB_DOWNLOAD_TIMEOUT (
    set "HF_HUB_DOWNLOAD_TIMEOUT=180"
)

echo ====================================
echo   VoxAI Studio
echo ====================================
echo.

:: Pre-startup cleanup to avoid port conflicts
echo [1/3] Checking for existing processes...

:: Check and kill any existing API server on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING" 2^>nul') do (
    echo     Found existing server on port 8000 ^(PID: %%a^), shutting down...
    curl -s -X POST http://127.0.0.1:8000/api/shutdown >nul 2>&1
    timeout /t 2 /nobreak >nul
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill any orphan electron processes from previous runs (but not other Electron apps)
for /f "tokens=2" %%i in ('tasklist /v /fi "imagename eq electron.exe" ^| findstr /i "VoxAI" 2^>nul') do (
    echo     Cleaning up orphan Electron process ^(PID: %%i^)...
    taskkill /F /PID %%i >nul 2>&1
)

echo [2/3] Environment ready.

cd electron

if not exist "node_modules" (
    echo [3/3] Installing dependencies...
    call "!NPM_CMD!" install
) else (
    echo [3/3] Dependencies OK.
)

echo.
echo Starting VoxAI Studio...
echo.

:: Start backend preheat in background (non-blocking)
start "" /B "%PYTHON_EXE%" "%~dp0preheat.py"

:: Immediately start Electron UI so the loading screen appears as soon as possible
call "!NPM_CMD!" start

pause

