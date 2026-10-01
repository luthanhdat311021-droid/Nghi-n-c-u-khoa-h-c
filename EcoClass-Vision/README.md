# EcoClass-Vision

Hệ thống phát hiện phòng học bị bỏ quên thiết bị điện dựa trên thị giác máy tính
(đề tài NCKH SV 2026–2027, Viện Công nghệ số – ĐH Thủ Dầu Một).

## Kiến trúc (theo Hình 1 – Hình 2 của thuyết minh)

| Tầng | Thành phần | Mã nguồn |
|---|---|---|
| Thu nhận | Camera RTSP/webcam/file, lấy mẫu 1–5 fps | `ecoclass/edge/capture.py` |
| | PZEM-004T + ESP32 (chỉ để tự gán nhãn/đối chứng) | `firmware/esp32_pzem/`, `ecoclass/data/sensor_log.py` |
| Xử lý tại biên | YOLO11n + ByteTrack → N(t), ngưỡng có trễ, cửa sổ trượt | `ecoclass/edge/person.py` |
| | Auto-ROI bằng YOLO-World (câu lệnh văn bản) | `ecoclass/edge/auto_roi.py`, `scripts/auto_roi.py` |
| | Trạng thái thiết bị: HSV (đèn), Farnebäck + độ nét (quạt), độ sáng + kết cấu (màn chiếu), LED (điều hòa), MobileNetV3 (tuỳ chọn) | `ecoclass/edge/devices.py` |
| | Máy trạng thái có trễ, E = Σ Pᵢ·Δtᵢ | `ecoclass/fsm.py` |
| | Làm mờ người, không lưu video | `ecoclass/edge/privacy.py` |
| | Sự kiện JSON → MQTT / HTTP | `ecoclass/edge/publisher.py` |
| Máy chủ | FastAPI + PostgreSQL/SQLite, MQTT bridge (Mosquitto) | `ecoclass/server/app.py`, `db.py`, `mqtt_bridge.py` |
| | Xác minh tầng 2 Florence-2 (chỉ khi có cảnh báo) | `ecoclass/server/verifier.py` |
| | Thời khóa biểu → chọn Te | `ecoclass/timetable.py`, `configs/timetable.csv` |
| Người dùng | Dashboard web, bot Telegram (/status /alerts /stats /ack) | `ecoclass/server/static/index.html`, `telegram_bot.py` |

## Cài đặt

```bash
python -m venv %USERPROFILE%\venvs\ecoclass
%USERPROFILE%\venvs\ecoclass\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
```

## Chạy demo đầu–cuối (không cần camera)

```bash
python scripts/make_demo_video.py --out data/demo
uvicorn ecoclass.server.app:app --port 8000          # cửa sổ 1 → mở http://127.0.0.1:8000
python scripts/run_edge.py --config configs/room_demo.yaml --show   # cửa sổ 2
```

Đánh giá mức sự kiện + mức khung, kèm ablation bỏ bộ suy luận thời gian:

```bash
python scripts/run_edge.py --config configs/room_demo.yaml --publisher none --start 2026-10-01T17:00:00 --frame-log runs/demo_frames.jsonl --events-out runs/demo_events.jsonl
python scripts/run_edge.py --config configs/room_demo.yaml --publisher none --start 2026-10-01T17:00:00 --no-temporal --events-out runs/ablation_no_temporal.jsonl
python scripts/evaluate_events.py --events runs/demo_events.jsonl runs/ablation_no_temporal.jsonl --gt data/demo/ground_truth.csv --start 2026-10-01T17:00:00 --te 20 --frame-log runs/demo_frames.jsonl --frame-gt data/demo/frame_gt.csv
```

## Quy trình triển khai cho phòng thật

1. **Auto-ROI**: `python scripts/auto_roi.py --source rtsp://... --room A1-101 --out configs/room_A1-101.yaml --preview roi.jpg` → rà soát/sửa ROI.
2. **Hiệu chuẩn**: `scripts/calibrate_devices.py --state off`, rồi `--state on`.
3. **Thu dữ liệu + tự gán nhãn**: `python -m ecoclass.data.collect ...`, `python -m ecoclass.data.sensor_log ...`, `python -m ecoclass.data.autolabel ... --person`.
4. **Huấn luyện**: `scripts/train_person.py` (YOLO11 tinh chỉnh, mAP/P/R), `scripts/train_device_cls.py --type light` (MobileNetV3 → ONNX, F1).
5. **Tối ưu cho Raspberry Pi 5**: `scripts/export_edge.py --formats ncnn onnx --imgsz 416`, đo FPS bằng `--benchmark`.
6. **Vận hành**: đặt `publisher.mode: mqtt`, chạy Mosquitto. Biến môi trường máy chủ (`.env`): `DATABASE_URL`, `MQTT_HOST`,
   `ECOCLASS_VLM=florence-2-base`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS`, `ECOCLASS_API_KEY`.

## Kiểm thử

```bash
pytest -q tests
```
