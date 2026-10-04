@echo off
title TruthPulse Fake News Detector
echo ========================================================
echo   Launching TruthPulse Fake News Detector...
echo   Opening in your web browser automatically!
echo ========================================================

cd /d "%~dp0"

:: Start the Streamlit app and open the browser
start "" http://localhost:8501
python -m streamlit run ui/app.py --server.port 8501 --server.headless false

pause
