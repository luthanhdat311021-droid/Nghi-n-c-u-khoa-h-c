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

## Huấn luyện từ video phòng học thật (2 phòng, 16 video)

Video đặt tại `datasets/classroom_videos/room{1,2}/`, nhãn theo video (đèn/quạt/người) ở `configs/classroom_labels.csv`.
Đèn dùng bộ phân lớp MobileNetV3 trên ảnh màu toàn khung; quạt dùng **ảnh chồng thời gian** (3 khung xám cách
nhau 0,1 s, căn chỉnh bù rung camera, xoá vùng người bằng mặt nạ YOLO11-seg — `ecoclass/edge/tstack.py`).

```bash
python scripts/build_scene_cls.py                                  # chia train/val theo video (4 video val)
python scripts/train_scene_cls.py --task light --out runs/scene_try
python scripts/train_scene_cls.py --task fan   --out runs/scene_try
python scripts/build_scene_cls.py --out datasets/scene_cls_all --val none   # bản cuối: toàn bộ video
python scripts/train_scene_cls.py --task light --data datasets/scene_cls_all  # → models/light_scene_mnv3.onnx
python scripts/train_scene_cls.py --task fan   --data datasets/scene_cls_all  # → models/fan_scene_mnv3.onnx
python scripts/eval_videos.py --config configs/room_classroom.yaml           # đánh giá đầu–cuối
python scripts/eval_videos.py --config configs/room_classroom_rules.yaml     # so sánh: luật thủ công
```

Kết quả (độ chính xác mức khung):

| | Luật thủ công (16 video) | Bộ phân lớp, khung val giữ riêng | Đầu–cuối trên 4 video val |
|---|---|---|---|
| Đèn | 0,487 | 1,000 (2 cách chia) | 1,000 |
| Quạt | 0,397 | 0,993 / 0,996 (2 cách chia) | 1,000 |
| Người | 1,000 | — | 1,000 |

Lưu ý: bộ phân lớp quạt cần pipeline chạy ≥ 10 fps (`cls_dt: 0.1`); dữ liệu mới chỉ có 2 phòng, phòng khác
nên quay thêm video (đủ tổ hợp đèn/quạt/người) rồi chạy lại các lệnh trên.

## Kiểm thử

```bash
pytest -q tests
```
