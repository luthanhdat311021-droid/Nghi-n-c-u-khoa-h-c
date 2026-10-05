@echo off
title EcoClass-Vision Edge Processing
echo ========================================================
echo   Running EcoClass-Vision Edge Processing (Demo Video)
echo ========================================================
if not exist "data\demo\demo_classroom.mp4" (
    echo [INFO] Video demo chưa tồn tại, đang khởi tạo video demo...
    python scripts\make_demo_video.py --out data\demo
)
echo [INFO] Đang chạy luồng phân tích thị giác máy tính Edge...
python scripts\run_edge.py --config configs\room_demo.yaml --source data\demo\demo_classroom.mp4 --show
pause
