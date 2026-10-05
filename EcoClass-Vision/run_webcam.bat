@echo off
title EcoClass-Vision Webcam
echo ========================================================
echo   EcoClass-Vision - Chạy với Webcam trực tiếp (Camera 0)
echo ========================================================
echo [INFO] Đang khởi chạy camera webcam... Nhấn ESC hoặc 'q' tại cửa sổ video để thoát.
python scripts\run_edge.py --config configs\room_demo.yaml --source 0 --show
pause
