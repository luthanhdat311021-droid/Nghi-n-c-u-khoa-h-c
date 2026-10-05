@echo off
title EcoClass-Vision Server
echo ========================================================
echo   Starting EcoClass-Vision Server ^& Dashboard
echo   Dashboard URL: http://127.0.0.1:8000
echo ========================================================
start "" http://127.0.0.1:8000
python -m uvicorn ecoclass.server.app:app --host 127.0.0.1 --port 8000
pause
