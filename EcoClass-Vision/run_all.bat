@echo off
title EcoClass-Vision One-Click Launcher
echo ========================================================
echo   EcoClass-Vision - Chạy hệ thống local tự động
echo ========================================================

if not exist "data\demo\demo_classroom.mp4" (
    echo [1/3] Đang tạo dữ liệu video demo...
    python scripts\make_demo_video.py --out data\demo
) else (
    echo [1/3] Video demo đã sẵn sàng.
)

echo [2/3] Đang khởi động Server FastAPI trên cổng 8000...
start "EcoClass-Vision Server" cmd /k "python -m uvicorn ecoclass.server.app:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8000

echo [3/3] Đang chạy Edge Computer Vision trên video demo...
python scripts\run_edge.py --config configs\room_demo.yaml --source data\demo\demo_classroom.mp4 --show

echo ========================================================
echo Hệ thống đã hoàn tất xử lý demo.
pause
