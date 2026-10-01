"""Tầng xác minh 2: mô hình thị giác – ngôn ngữ Florence-2.

Chỉ được gọi khi thiết bị biên phát cảnh báo (sự kiện nghi ngờ), không chạy liên tục.
Nhiệm vụ: kiểm tra lại ảnh bằng chứng xem còn người mà YOLO bỏ sót không
(gục trên bàn, ngồi góc khuất...). Nếu có → cảnh báo bị loại (rejected) và vùng
người do Florence-2 tìm thấy được làm mờ trước khi lưu ảnh.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np

PERSON_WORDS = re.compile(r"\b(person|people|man|men|woman|women|boy|girl|student|students|child|human|teacher)\b", re.I)


@dataclass
class VerifyResult:
    person_present: bool
    person_boxes: list[list[float]] = field(default_factory=list)
    caption: str = ""
    labels: list[str] = field(default_factory=list)


class Florence2Verifier:
    HF_IDS = {
        "florence-2-base": ["florence-community/Florence-2-base", "microsoft/Florence-2-base"],
        "florence-2-large": ["florence-community/Florence-2-large", "microsoft/Florence-2-large"],
    }

    def __init__(self, variant: str = "florence-2-base", device: str = "auto"):
        self.variant = variant
        self.device_pref = device
        self._lock = threading.Lock()
        self.model = self.processor = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device_pref == "auto" else self.device_pref
        dtype = torch.float16 if dev.startswith("cuda") else torch.float32
        last_err = None
        for hf_id in self.HF_IDS[self.variant]:
            try:
                if hf_id.startswith("florence-community"):
                    # transformers >= 4.56 hỗ trợ Florence-2 gốc, không cần trust_remote_code
                    from transformers import Florence2ForConditionalGeneration
                    model = Florence2ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype)
                    processor = AutoProcessor.from_pretrained(hf_id)
                else:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, trust_remote_code=True)
                    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                self.model, self.processor, self.device, self.dtype = model.to(dev).eval(), processor, dev, dtype
                print(f"[verifier] đã nạp {hf_id} trên {dev}")
                return
            except Exception as e:  # thử checkpoint tiếp theo
                last_err = e
        raise RuntimeError(f"Không nạp được Florence-2: {last_err}")

    def _run(self, image, task: str, text: str = "") -> dict:
        import torch

        inputs = self.processor(text=task + text, images=image, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.no_grad():
            ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                      max_new_tokens=256, num_beams=3, do_sample=False)
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(raw, task=task, image_size=image.size)

    def verify(self, bgr: np.ndarray) -> VerifyResult:
        from PIL import Image

        with self._lock:
            self._load()
            image = Image.fromarray(bgr[..., ::-1].copy())
            od = self._run(image, "<OD>").get("<OD>", {})
            cap = self._run(image, "<MORE_DETAILED_CAPTION>").get("<MORE_DETAILED_CAPTION>", "")
        labels = list(od.get("labels", []))
        boxes = [b for b, l in zip(od.get("bboxes", []), labels) if PERSON_WORDS.search(l)]
        # Chú thích chỉ là tín hiệu phụ: chỉ kết luận có người khi OD tìm được box người
        # hoặc chú thích nói rõ có người đang ngồi/ngủ trong phòng.
        cap_person = bool(PERSON_WORDS.search(cap)) and not re.search(r"\b(no|without|empty of) (people|person|one)\b", cap, re.I)
        return VerifyResult(person_present=bool(boxes) or cap_person, person_boxes=boxes, caption=str(cap), labels=labels)


class NoopVerifier:
    def verify(self, bgr):
        return None


def make_verifier(name: str, device: str = "auto"):
    if name in ("off", "", None):
        return NoopVerifier()
    return Florence2Verifier(name, device)
