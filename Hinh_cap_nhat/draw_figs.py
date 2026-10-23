# -*- coding: utf-8 -*-
"""Vẽ lại Hình 1 (kiến trúc) và Hình 2 (máy trạng thái) theo hiện trạng EcoClass-Vision."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
INK = "#333333"


def canvas(w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def tier(ax, x, y, w, h, color, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", fc=color,
                                ec="#9a9a9a", lw=1.6, ls=(0, (4, 3))))
    ax.text(x + w - 22, y + 30, title, ha="right", va="center", fontsize=16, weight="bold", color="#555555")


def box(ax, x, y, w, h, text, fc="white", bold=False, fs=15.5, ls="-", ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12", fc=fc, ec=ec, lw=2.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            color="black", linespacing=1.35)


def arrow(ax, p, q, color=INK, ls="-", lw=2.2, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=22,
                                                    shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def poly_arrow(ax, pts, color=INK, ls="-", lw=2.2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, ls=ls, lw=lw, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def label(ax, x, y, text, fs=14, color="black", ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.3, style=style)


RED = "#c0392b"
GREY = "#777777"

# ------------------------------------------------------------------ Hình 1
fig, ax = canvas(1600, 1520)
tier(ax, 30, 20, 1540, 300, "#eaf1fb", "TẦNG THU NHẬN VÀ IoT")
tier(ax, 30, 340, 1540, 525, "#edf6ec", "TẦNG XỬ LÝ TẠI BIÊN (Pi 5 / máy tính GPU)")
tier(ax, 30, 880, 1540, 250, "#fdf3e4", "TẦNG MÁY CHỦ")
tier(ax, 30, 1150, 1540, 280, "#f3edf7", "TẦNG NGƯỜI DÙNG")

# Tầng thu nhận + IoT
box(ax, 60, 70, 460, 120, "Camera phòng học (IP/RTSP)\nImou Ranger 2 có hồng ngoại ban đêm;\nwebcam, điện thoại khi thử nghiệm")
box(ax, 545, 70, 440, 120, "ESP32 + cảm biến DHT11\nnhiệt độ, độ ẩm → trạng thái điều hòa\n(USB hoặc Wi-Fi)")
box(ax, 1005, 70, 535, 120, "Cảm biến điện năng PZEM-004T + ESP32\n(chỉ dùng để tự gán nhãn và đối chứng)",
    ls=(0, (5, 3)), ec=GREY)
box(ax, 1030, 215, 220, 80, "Máy lạnh\n(điều hòa)", fc="#f7f7f7")
box(ax, 1280, 215, 260, 80, "ESP32 + LED\nphát hồng ngoại", fc="#fdecea")
arrow(ax, (1280, 255), (1250, 255), color=RED)

# Tầng xử lý tại biên
box(ax, 80, 400, 1440, 62, "Lấy mẫu khung hình 5–15 fps (luồng đọc riêng, luôn giữ khung mới nhất), tiền xử lý, đồng bộ thời gian")
arrow(ax, (290, 190), (290, 400))
arrow(ax, (765, 190), (765, 400))
arrow(ax, (1017, 190), (1017, 400), color=GREY, ls=(0, (5, 3)))
box(ax, 80, 500, 440, 130, "Phát hiện người YOLO11n-seg\n+ ByteTrack → số người N(t),\nmặt nạ người, làm trơn cửa sổ trượt")
box(ax, 545, 500, 440, 130, "Định vị thiết bị: Auto-ROI YOLO-World;\nYOLO11 chỉ cho loại quạt trần\n(nhãn tự sinh bằng YOLO-World)")
box(ax, 1010, 490, 510, 170, "Trạng thái thiết bị:\nĐèn – MobileNetV3 toàn khung + vùng sáng\nQuạt – dao động độ sáng trong hộp quạt\nĐiều hòa – nhiệt độ DHT11", fs=15)
for x in (300, 765, 1265):
    arrow(ax, (x, 462), (x, 490 if x == 1265 else 500))
arrow(ax, (985, 565), (1010, 565))
box(ax, 80, 715, 1170, 120, "Bộ suy luận theo thời gian (máy trạng thái có trễ)\n+ làm mờ người → sự kiện JSON, ảnh trực tiếp đã làm mờ",
    fc="#dcefd8", bold=True, fs=16.5)
box(ax, 1280, 715, 240, 120, "Thời khóa biểu\nphòng học\n→ chọn T$_e$")
arrow(ax, (1280, 775), (1250, 775))
arrow(ax, (300, 630), (300, 715))
arrow(ax, (1265, 660), (1165, 715))

# Tầng máy chủ
box(ax, 60, 940, 260, 110, "MQTT (Mosquitto)\nhoặc HTTP")
box(ax, 350, 940, 360, 110, "Backend FastAPI\n+ PostgreSQL / SQLite")
box(ax, 740, 940, 300, 110, "Xác minh tầng 2\nFlorence-2 (VLM)")
box(ax, 1070, 940, 450, 110, "Điều khiển IoT: gửi lệnh tắt điều hòa\nđến ESP32 phát hồng ngoại", fc="#fdecea")
arrow(ax, (190, 835), (190, 940))
arrow(ax, (320, 995), (350, 995))
arrow(ax, (710, 985), (740, 985))
arrow(ax, (740, 1005), (710, 1005))
poly_arrow(ax, [(640, 1050), (640, 1090), (1295, 1090), (1295, 1050)], color=RED)
poly_arrow(ax, [(1520, 995), (1555, 995), (1555, 255), (1540, 255)], color=RED)

# Tầng người dùng
box(ax, 50, 1210, 520, 130, "Dashboard web: trạng thái phòng,\ncamera trực tiếp, nhiệt độ, thống kê kWh,\nnút tắt điều hòa")
box(ax, 595, 1210, 300, 130, "Telegram Bot\n(cảnh báo kèm\nảnh bằng chứng)")
box(ax, 920, 1210, 300, 130, "Chatbot Zalo\n(cảnh báo, tra cứu,\nlệnh tắt điều hòa)")
box(ax, 1245, 1210, 300, 130, "Bảo vệ / Quản lý\nkiểm tra, xử lý")
def both(p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2.2, mutation_scale=22,
                                                    shrinkA=0, shrinkB=0))


both((430, 1050), (300, 1210))
arrow(ax, (560, 1050), (745, 1210))
both((660, 1050), (1030, 1210))
arrow(ax, (1220, 1275), (1245, 1275))

# Chú giải
ly = 1475
ax.plot([45, 100], [ly, ly], color=INK, lw=2.2)
label(ax, 108, ly, "dữ liệu, sự kiện", ha="left")
ax.annotate("", xy=(375, ly), xytext=(320, ly),
            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2.2, mutation_scale=18, shrinkA=0, shrinkB=0))
label(ax, 383, ly, "cảnh báo xuống, lệnh tắt điều hòa lên", ha="left")
ax.plot([840, 895], [ly, ly], color=RED, lw=2.2)
label(ax, 903, ly, "thực thi lệnh qua hồng ngoại", ha="left")
ax.plot([1190, 1245], [ly, ly], color=GREY, lw=2.2, ls=(0, (5, 3)))
label(ax, 1253, ly, "chỉ để gán nhãn, đối chứng", ha="left")
fig.savefig(os.path.join(OUT, "hinh1_kien_truc.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Hình 2
fig, ax = canvas(1840, 800)
S = {"OCC": (40, 200, 310, 140, "CÓ NGƯỜI\n(OCCUPIED)", "#eaf1fb"),
     "PEND": (515, 200, 310, 140, "NGHI VẮNG\n(EMPTY_PENDING)", "#fdf3e4"),
     "ALERT": (985, 200, 320, 140, "BỎ QUÊN THIẾT BỊ\n(ALERT)", "#f9dedc"),
     "RES": (1480, 200, 310, 140, "ĐÃ XỬ LÝ\n(RESOLVED)", "#edf6ec"),
     "SAFE": (515, 560, 310, 140, "TRỐNG – AN TOÀN\n(mọi thiết bị tắt)", "#edf6ec")}
for x, y, w, h, t, c in S.values():
    box(ax, x, y, w, h, t, fc=c, bold=True, fs=19)
# CÓ NGƯỜI ↔ NGHI VẮNG
arrow(ax, (350, 245), (515, 245))
label(ax, 432, 205, "N(t)=0\nliên tục ≥ t$_p$", fs=15)
arrow(ax, (515, 300), (350, 300))
label(ax, 432, 345, "N(t)>0\n(xác nhận)", fs=15)
# NGHI VẮNG → BỎ QUÊN
arrow(ax, (825, 270), (985, 270))
label(ax, 905, 335, "vắng ≥ T$_e$ và\ncó thiết bị\nđang BẬT", fs=15)
# BỎ QUÊN → ĐÃ XỬ LÝ
arrow(ax, (1305, 270), (1480, 270))
label(ax, 1392, 340, "mọi thiết bị TẮT\nổn định ≥ t$_{off}$\n(tắt tại chỗ hoặc\ntừ xa qua dashboard,\nZalo → hồng ngoại)", fs=14)
# NGHI VẮNG → TRỐNG
arrow(ax, (670, 340), (670, 560))
label(ax, 650, 445, "vắng ≥ T$_e$ và\nmọi thiết bị TẮT", fs=15, ha="right")
# TRỐNG → CÓ NGƯỜI
arrow(ax, (515, 590), (200, 340))
label(ax, 290, 495, "N(t)>0", fs=15)
# BỎ QUÊN ⇢ CÓ NGƯỜI (đóng cảnh báo)
ax.annotate("", xy=(205, 200), xytext=(1145, 200),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.2, ls=(0, (5, 3)), mutation_scale=22,
                            connectionstyle="arc3,rad=0.25", shrinkA=0, shrinkB=0))
label(ax, 675, 70, "có người quay lại, N(t)>0: đóng cảnh báo", fs=16, color="#555555")
# ĐÃ XỬ LÝ → TRỐNG
arrow(ax, (1635, 340), (825, 650))
label(ax, 1330, 560, "tức thời: lưu thời gian lãng phí,\nước tính kWh", fs=15)
# TRỐNG → BỎ QUÊN (mới)
arrow(ax, (825, 590), (1080, 340), color=RED)
label(ax, 925, 455, "thiết bị BẬT lại khi\nphòng vẫn trống", fs=15, color=RED, ha="right")
label(ax, 920, 765, "t$_p$: thời gian không thấy người để chuyển sang nghi vắng;  T$_e$: ngưỡng vắng (theo thời khóa biểu);  "
      "t$_{off}$: thời gian xác nhận thiết bị đã tắt", fs=14, color="#555555")
fig.savefig(os.path.join(OUT, "hinh2_may_trang_thai.png"), dpi=200)
plt.close(fig)
print("done")
