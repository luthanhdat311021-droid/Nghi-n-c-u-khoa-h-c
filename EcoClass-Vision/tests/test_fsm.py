from ecoclass.config import FSMConfig
from ecoclass.fsm import OFF, ON, RoomState, TemporalFSM

CFG = FSMConfig(empty_seconds_to_pending=2, te_seconds=10, te_no_class_seconds=10, off_confirm_seconds=2)
POWER = {"light": 72.0, "fan": 60.0}


def run(fsm, seq):
    """seq: list of (t_from, t_to, people, devices); bước 0.5 s."""
    events = []
    for t0, t1, n, dev in seq:
        t = t0
        while t < t1:
            events += fsm.update(t, n, dev)
            t = round(t + 0.5, 3)
    return events


def types(events):
    return [(e.type, e.state_to) for e in events if e.type != "state_change"]


def test_forgotten_devices_alert_then_resolved():
    fsm = TemporalFSM("R1", CFG, POWER)
    ev = run(fsm, [
        (0, 5, 2, {"light": ON, "fan": ON}),
        (5, 30, 0, {"light": ON, "fan": ON}),
        (30, 40, 0, {"light": OFF, "fan": OFF}),
    ])
    alerts = [e for e in ev if e.type == "alert"]
    assert len(alerts) == 1
    # Người cuối cùng thấy ở t=4.5 → cảnh báo khi vắng >= Te=10 s
    assert 14.0 <= alerts[0].ts <= 15.5
    assert set(alerts[0].devices_on) == {"fan", "light"}
    closed = [e for e in ev if e.type == "alert_closed"]
    assert len(closed) == 1 and closed[0].reason == "devices_off"
    # Lãng phí ≈ (72+60) W × ~25.5 s
    assert 0.8 < closed[0].waste_wh < 1.05
    assert fsm.state == RoomState.EMPTY_SAFE


def test_empty_room_all_off_is_safe():
    fsm = TemporalFSM("R1", CFG, POWER)
    ev = run(fsm, [(0, 3, 1, {"light": OFF}), (3, 20, 0, {"light": OFF})])
    assert not [e for e in ev if e.type == "alert"]
    assert fsm.state == RoomState.EMPTY_SAFE


def test_short_absence_no_alert():
    """Sinh viên ra ngoài ít hơn Te rồi quay lại: không cảnh báo."""
    fsm = TemporalFSM("R1", CFG, POWER)
    ev = run(fsm, [(0, 5, 3, {"light": ON}), (5, 12, 0, {"light": ON}), (12, 20, 2, {"light": ON})])
    assert not [e for e in ev if e.type == "alert"]
    assert fsm.state == RoomState.OCCUPIED


def test_single_frame_false_person_does_not_cancel():
    fsm = TemporalFSM("R1", CFG, POWER)
    run(fsm, [(0, 2, 1, {"light": ON}), (2, 20, 0, {"light": ON})])
    assert fsm.state == RoomState.ALERT
    fsm.update(20.0, 1, {"light": ON})     # một khung nhiễu
    fsm.update(20.5, 0, {"light": ON})
    assert fsm.state == RoomState.ALERT


def test_person_returns_cancels_alert():
    fsm = TemporalFSM("R1", CFG, POWER)
    ev = run(fsm, [(0, 2, 1, {"light": ON}), (2, 20, 0, {"light": ON}), (20, 22, 1, {"light": ON})])
    closed = [e for e in ev if e.type == "alert_closed"]
    assert closed and closed[0].reason == "person_returned"
    assert fsm.state == RoomState.OCCUPIED


def test_device_turns_on_while_empty():
    fsm = TemporalFSM("R1", CFG, POWER)
    run(fsm, [(0, 2, 1, {"light": OFF}), (2, 20, 0, {"light": OFF})])
    assert fsm.state == RoomState.EMPTY_SAFE
    ev = fsm.update(20.0, 0, {"light": ON})
    assert [e.type for e in ev if e.type == "alert"] == ["alert"]


def test_te_depends_on_timetable():
    fsm = TemporalFSM("R1", CFG, POWER, te_fn=lambda ts: 4 if ts > 100 else 10)
    run(fsm, [(100, 102, 1, {"light": ON}), (102, 110, 0, {"light": ON})])
    assert fsm.state == RoomState.ALERT
