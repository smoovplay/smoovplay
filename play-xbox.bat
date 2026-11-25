@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ============================================================================
:: SmoovPlay + Native Xbox One-Click Launcher
:: Double-click to build (if needed), sign in (first run), start Xbox Remote
:: Play, and launch SmoovPlay in low-delay DXGI fullscreen.
::
:: Optional overrides (edit here or set in environment before running):
::   XBOX_RES      = 1080p | 1440p | 720p
::   XBOX_BITRATE  = max video bitrate in kbps (default 20000)
::   XBOX_MODE     = --home  (or --home "Console Name", or --cloud <titleId>)
::   SMOOVPLAY_PYTHON = path to Python executable (defaults to repo venv or installed SmoovPlay runtime)
:: ============================================================================
if not defined XBOX_RES set "XBOX_RES=1080p"
if not defined XBOX_BITRATE set "XBOX_BITRATE=20000"
if not defined XBOX_MODE set "XBOX_MODE=--home"
if not defined SMOOVPLAY_ARGS set "SMOOVPLAY_ARGS=--source tap --display --present dxgi --fullscreen"

set "XBOX_DIR=%~dp0xbox_tap"
set "XBOX_EXE=%XBOX_DIR%\smoovplay-xbox.exe"
set "TOKEN_FILE=%LOCALAPPDATA%\SmoovPlay\xbox_tokens.json"

:: 1. Locate Python (prefer SMOOVPLAY_PYTHON, repo venv, then installed SmoovPlay runtime)
set "PY_CMD="
set "PY_EXE="
set "PY_EXTRA="
if defined SMOOVPLAY_PYTHON if exist "%SMOOVPLAY_PYTHON%" (
    set "PY_EXE=%SMOOVPLAY_PYTHON%"
    set "PY_CMD="%SMOOVPLAY_PYTHON%""
)
if not defined PY_CMD if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
    set "PY_CMD="%~dp0.venv\Scripts\python.exe""
)
if not defined PY_CMD if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0venv\Scripts\python.exe"
    set "PY_CMD="%~dp0venv\Scripts\python.exe""
)
if not defined PY_CMD (
    for %%D in (
        "%LOCALAPPDATA%\Programs\SmoovPlay"
        "%LOCALAPPDATA%\SmoovPlay"
        "D:\SmoovPlay Xbox"
        "D:\SmoovPlay"
        "C:\SmoovPlay"
    ) do (
        if not defined PY_CMD if exist "%%~D\runtime\python\python.exe" (
            set "PY_EXE=%%~D\runtime\python\python.exe"
            set "PY_CMD="%%~D\runtime\python\python.exe""
            if not defined FRAMEGEN_VFX_PATH if exist "%%~D\runtime\vfx" (
                set "FRAMEGEN_VFX_PATH=%%~D\runtime\vfx"
            )
        )
    )
)
if not defined PY_CMD (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_EXE=python"
        set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_EXE=py"
        set "PY_EXTRA=-3"
        set "PY_CMD=py -3"
    )
)
if not defined PY_CMD (
    echo [error] Python not found. Activate your venv, set SMOOVPLAY_PYTHON, or install Python 3.
    pause
    exit /b 1
)
%PY_CMD% -c "import cv2, torch" >nul 2>&1
if !errorlevel! neq 0 (
    echo [error] Python (!PY_CMD!^) is missing required packages ^(cv2, torch^).
    echo         Set SMOOVPLAY_PYTHON to your SmoovPlay Python or run: !PY_CMD! -m pip install -r requirements.txt
    pause
    exit /b 1
)

:: 2. Build/update smoovplay-xbox.exe automatically (Go caches unchanged builds)
set "GO_CMD="
where go >nul 2>&1
if %errorlevel% equ 0 (
    set "GO_CMD=go"
) else if exist "C:\Program Files\Go\bin\go.exe" (
    set "GO_CMD="C:\Program Files\Go\bin\go.exe""
)
if defined GO_CMD (
    echo [xbox] Building smoovplay-xbox.exe...
    pushd "%XBOX_DIR%"
    %GO_CMD% build -o smoovplay-xbox.exe ./cmd/smoovplay-xbox
    set "BUILD_ERR=!errorlevel!"
    popd
    if !BUILD_ERR! neq 0 (
        echo [error] go build failed.
        pause
        exit /b !BUILD_ERR!
    )
) else if not exist "%XBOX_EXE%" (
    echo [error] "%XBOX_EXE%" not found and Go is not installed.
    echo         Install Go with: winget install GoLang.Go
    pause
    exit /b 1
)

:: 3. First-time Microsoft Device Code sign-in if no cached tokens exist
if not exist "%TOKEN_FILE%" (
    echo [xbox] No saved Xbox login found. Starting one-time sign-in...
    "%XBOX_EXE%" login
    if !errorlevel! neq 0 (
        echo [error] Xbox sign-in failed.
        pause
        exit /b 1
    )
)

:: 4. Stop any leftover smoovplay-xbox instance before starting fresh
taskkill /F /IM smoovplay-xbox.exe >nul 2>&1

:: 5. Start native Xbox WebRTC stream in a minimized companion console (logged to out\xbox-stream.log)
if not exist "%~dp0out" mkdir "%~dp0out"
set "XBOX_LOG=%~dp0out\xbox-stream.log"
set "SMOOVPLAY_LOG=%~dp0out\smoovplay-run.log"
set "SMOOVPLAY_CSV=%~dp0out\smoovplay-ticks.csv"
set "PYTHONFAULTHANDLER=1"
echo [xbox] Starting native Xbox stream (%XBOX_MODE%, %XBOX_RES%, %XBOX_BITRATE% kbps)...
start "SmoovPlay Xbox Stream" /min cmd /c ""%XBOX_EXE%" play %XBOX_MODE% --res %XBOX_RES% --bitrate %XBOX_BITRATE% --stats 1s > "%XBOX_LOG%" 2>&1"

:: 6. Launch SmoovPlay in the foreground (waits for stream on Local\smoovplay-tap)
echo [smoovplay] Launching SmoovPlay (%SMOOVPLAY_ARGS% %*)...
powershell -NoProfile -Command "$ErrorActionPreference='Continue'; & $env:PY_EXE %PY_EXTRA% -u '%~dp0run.py' %SMOOVPLAY_ARGS% --chiaki-log '%SMOOVPLAY_CSV%' %* 2>&1 | Tee-Object -FilePath '%SMOOVPLAY_LOG%'; exit $LASTEXITCODE"
set "RUN_ERR=%errorlevel%"

:: 7. Cleanly stop the Xbox stream when SmoovPlay closes
echo [xbox] Stopping Xbox stream...
taskkill /IM smoovplay-xbox.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul
taskkill /F /IM smoovplay-xbox.exe /T >nul 2>&1

if %RUN_ERR% neq 0 (
    echo [smoovplay] Exited with code %RUN_ERR%.
    pause
    exit /b %RUN_ERR%
)
exit /b 0
