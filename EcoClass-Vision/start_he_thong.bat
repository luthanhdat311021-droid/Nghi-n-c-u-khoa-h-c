@echo off
chcp 65001 >nul
REM Bấm đúp để chạy hệ thống quản lý phòng học (camera + ESP32 + dashboard http://127.0.0.1:8000).
REM Đóng cửa sổ này hoặc nhấn Ctrl+C để tắt.
cd /d "%~dp0"
start "" http://127.0.0.1:8000
"%USERPROFILE%\venvs\ecoclass\Scripts\python.exe" scripts\run_system.py %*
pause
