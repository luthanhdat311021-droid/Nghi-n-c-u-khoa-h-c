import numpy as np

from ecoclass.config import DeviceConfig, FSMConfig
from ecoclass.edge.devices import build_detector, make_scene
from ecoclass.edge.esp32 import SensorReading, parse_line
from ecoclass.fsm import OFF, ON, UNKNOWN, RoomState, TemporalFSM

FRAME = np.zeros((48, 64, 3), np.uint8)


def test_parse_firmware_line():
    r = parse_line("STA_STATUS=3 STA_IP=192.168.221.101 AP_CLIENTS=0 Temperature_C=27.5 Humidity_percent=56.4",
                   SensorReading(), 100.0)
    assert (r.temperature, r.humidity, r.ts, r.esp_ip) == (27.5, 56.4, 100.0, "192.168.221.101")


def test_parse_read_error_keeps_last_value():
    prev = SensorReading(27.5, 56.4, 100.0, "1.2.3.4")
    r = parse_line("STA_STATUS=6 STA_IP=0.0.0.0 AP_CLIENTS=0 DHT_READ_ERROR", prev, 101.0)
    assert r.error and r.temperature == 27.5 and r.ts == 100.0      # ts cũ → sẽ bị coi là cũ (stale)
    assert parse_line("ets Jul 29 2019 12:21:46", prev, 102.0) == prev   # dòng khởi động: bỏ qua


def ac_detector(**params):
    return build_detector(DeviceConfig(id="ac", type="ac_temp", roi=[0, 0, 1, 1], on_frames=3, off_frames=3,
                                       params=params))


def feed(det, temps):
    out = None
    for i, t in enumerate(temps):
        out = det.update(FRAME, make_scene(FRAME, [], sensors={"temperature": t}), i * 0.1)
    return out


def test_ac_on_below_30():
    assert feed(ac_detector(), [27.5] * 3).state == ON
    assert feed(ac_detector(), [31.0] * 3).state == OFF


def test_ac_hysteresis_and_debounce():
    det = ac_detector()
    feed(det, [29.0] * 3)
    assert feed(det, [30.2] * 5).state == ON          # trong vùng trễ 30–30,5 → giữ BẬT
    assert feed(det, [31.0, 29.0, 31.0]).state == ON  # nhiễu 1 lần đọc không lật trạng thái
    assert feed(det, [31.0] * 3).state == OFF


def test_ac_unknown_without_sensor():
    det = ac_detector()
    r = det.update(FRAME, make_scene(FRAME, [], sensors={"temperature": None}), 0.0)
    assert r.raw == UNKNOWN and r.state == UNKNOWN
    assert det.update(FRAME, make_scene(FRAME, []), 0.1).state == UNKNOWN


def run_fsm(fsm, seq):
    events = []
    for t0, t1, n, dev in seq:
        t = t0
        while t < t1:
            events += fsm.update(t, n, dev)
            t += 1
    return events


CFG = FSMConfig(empty_seconds_to_pending=3, te_seconds=120, te_no_class_seconds=120, off_confirm_seconds=5)
POWER = {"lights": 72.0, "fans": 60.0, "ac": 1000.0}


def test_alert_after_2_minutes_when_only_ac_cooling():
    fsm = TemporalFSM("C204", CFG, POWER)
    dev = {"lights": OFF, "fans": OFF, "ac": ON}
    ev = run_fsm(fsm, [(0, 10, 2, dev), (10, 200, 0, dev)])
    alerts = [e for e in ev if e.type == "alert"]
    assert len(alerts) == 1 and alerts[0].devices_on == ["ac"]
    assert 129 <= alerts[0].ts <= 131                  # người cuối thấy ở t=9 → cảnh báo sau 2 phút


def test_no_alert_before_2_minutes():
    fsm = TemporalFSM("C204", CFG, POWER)
    dev = {"lights": ON, "fans": OFF, "ac": OFF}
    ev = run_fsm(fsm, [(0, 10, 2, dev), (10, 120, 0, dev), (120, 130, 1, dev)])
    assert not [e for e in ev if e.type == "alert"]
    assert fsm.state == RoomState.OCCUPIED


def test_empty_room_warm_and_all_off_is_safe():
    fsm = TemporalFSM("C204", CFG, POWER)
    dev = {"lights": OFF, "fans": OFF, "ac": OFF}
    ev = run_fsm(fsm, [(0, 10, 1, dev), (10, 300, 0, dev)])
    assert not [e for e in ev if e.type == "alert"]
    assert fsm.state == RoomState.EMPTY_SAFE
