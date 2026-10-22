@echo off
REM ============================================================
REM  run.bat — Launch the ALPR Streamlit application on Windows
REM  Usage: Double-click or run from project root
REM  Author: Karam Singh Bana (Team Member 2)
REM ============================================================

echo.
echo =====================================================
echo  Vehicle Number Plate Recognition System
echo  AI-powered ALPR — College Engineering Project
echo =====================================================
echo.

REM Check Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python is not found in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install or update dependencies
echo [INFO] Installing/verifying dependencies...
pip install -r requirements.txt --quiet
IF ERRORLEVEL 1 (
    echo [ERROR] Dependency installation failed. Check requirements.txt.
    pause
    exit /b 1
)

REM Copy .env.example to .env if .env does not exist
IF NOT EXIST ".env" (
    IF EXIST ".env.example" (
        copy ".env.example" ".env" >nul
        echo [INFO] Created .env from .env.example — review and set ALPR_MODEL_PATH.
    )
)

echo.
echo [INFO] Starting Streamlit application...
echo [INFO] Open your browser at http://localhost:8501
echo [INFO] Press Ctrl+C to stop the server.
echo.

streamlit run app.py

pause
