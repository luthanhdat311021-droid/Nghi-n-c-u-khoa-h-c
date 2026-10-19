from ecoclass.server import zalo_bot
from ecoclass.server.zalo_bot import ZaloBot, html_to_text, parse_updates, split_text


def _msg(cid, text, is_bot=False):
    return {"message": {"from": {"id": cid, "is_bot": is_bot}, "chat": {"id": cid, "chat_type": "PRIVATE"},
                        "text": text, "message_id": "m1", "date": 1750316131602},
            "event_name": "message.text.received"}


def test_html_to_text():
    assert html_to_text("⚠️ <b>Phòng C204</b>\n/ack &lt;id&gt; &amp; xong") == "⚠️ Phòng C204\n/ack <id> & xong"


def test_split_text_respects_limit():
    s = "\n".join(["x" * 30] * 10) + "\n" + "y" * 120
    parts = split_text(s, limit=100)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(parts).replace("\n", "") == s.replace("\n", "")


def test_parse_updates_single_and_list():
    assert parse_updates({"ok": True, "result": _msg("abc", " /status ")}) == [("abc", "/status")]
    assert parse_updates({"ok": True, "result": [_msg("a", "hi"), _msg("b", "x", is_bot=True)]}) == [("a", "hi")]
    assert parse_updates({"ok": False, "error_code": 408}) == []
    assert parse_updates({"ok": True, "result": {"event_name": "message.sticker.received", "message": {}}}) == []


def test_reply_unknown_chat_gets_its_id():
    bot = ZaloBot("1:t", ["allowed"], {"status": lambda a, c: "ok"})
    r = bot.reply("stranger", "/status")
    assert "stranger" in r and "ZALO_CHAT_IDS" in r


def test_reply_commands():
    seen = {}

    def ack(arg, cid):
        seen["ack"] = (arg, cid)
        return "đã xác nhận"

    bot = ZaloBot("1:t", ["me"], {"status": lambda a, c: "<b>C204</b>", "ack": ack})
    assert bot.reply("me", "/status") == "<b>C204</b>"
    assert bot.reply("me", "/ACK a1") == "đã xác nhận" and seen["ack"] == ("a1", "zalo:me")
    assert bot.reply("me", "xin chào") == zalo_bot.HELP
    assert bot.reply("me", "/foo") == zalo_bot.HELP


def test_send_text_posts_plain_text(monkeypatch):
    calls = []

    class R:
        def json(self):
            return {"ok": True, "result": {}}

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return R()

    monkeypatch.setattr(zalo_bot.requests, "post", fake_post)
    bot = ZaloBot("1:t", ["c1", "c2"], api_base="https://api.test/")
    assert bot.send_text("⚠️ <b>Phòng C204</b>") is True
    assert calls == [("https://api.test/bot1:t/sendMessage", {"chat_id": c, "text": "⚠️ Phòng C204"}) for c in ("c1", "c2")]


def test_disabled_without_chat_ids(monkeypatch):
    monkeypatch.setattr(zalo_bot.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    assert ZaloBot("1:t", []).send_text("x") is False
    assert ZaloBot(None, ["c"]).enabled is False
