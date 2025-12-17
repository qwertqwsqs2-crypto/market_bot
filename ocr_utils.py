from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from models import Rect


_PRICE_RE = re.compile(r"(\d[\d\s,\.]*)")


def parse_price_to_int(text: str) -> Optional[int]:
    """
    Normalize typical UI price formats:
      '5', '1 700 000', '2,299,999', '3 333 333', '5 001', etc.
    - extract digit group
    - remove spaces and separators
    """
    if not text:
        return None
    text = text.replace("\u00A0", " ")  # NBSP
    text = re.sub(r"(?<=\d)[lI|](?=\s*\d{3}\b)", "1", text)
    m = _PRICE_RE.search(text)
    if not m:
        return None
    s = m.group(1)
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


@dataclass(frozen=True)
class OCRResult:
    value: Optional[int]
    confidence: float
    raw_text: str
    engine: str


class OCRReader:
    """
    OCR strategy:
      - preprocess crop in a few variants (scale/threshold/morphology)
      - run easyocr first (if enabled)
      - fallback to tesseract (optional)
      - pick best result by confidence and/or majority vote
    """
    def __init__(self, use_easyocr: bool, langs: Sequence[str], use_tesseract: bool, min_conf: float, prefer_gpu: bool = True) -> None:
        self.use_easyocr = use_easyocr
        self.use_tesseract = use_tesseract
        self.min_conf = min_conf

        self._easy_reader = None
        if self.use_easyocr:
            try:
                import easyocr  # type: ignore
                gpu_flag = prefer_gpu and self._gpu_available()
                self._easy_reader = easyocr.Reader(list(langs), gpu=gpu_flag)
                if gpu_flag and getattr(self._easy_reader, "device", "cpu") == "cpu":
                    logging.getLogger(__name__).warning("easyocr GPU requested but CPU was selected; check CUDA setup")
            except Exception:
                self._easy_reader = None
                self.use_easyocr = False

        self._tesseract_ok = False
        if self.use_tesseract:
            try:
                import pytesseract  # type: ignore  # noqa: F401
                self._tesseract_ok = True
            except Exception:
                self._tesseract_ok = False
                self.use_tesseract = False

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _gold_mask_variant(img_bgr: np.ndarray, scale: int) -> np.ndarray:
        import cv2  # type: ignore

        bgr = img_bgr
        if scale > 1:
            bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # диапазон под "золотые" цифры (можешь чуть подвигать H/S/V при необходимости)
        lower = np.array([10, 70, 70], dtype=np.uint8)
        upper = np.array([45, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # важно: НЕ делай aggressive OPEN 2x2/3x3 — он ломает тонкие штрихи "3"
        k = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # easyocr лучше ест белое на чёрном
        return mask

    @staticmethod
    def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
        import cv2  # type: ignore

        if len(img_bgr.shape) == 2:
            return img_bgr
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def preprocess_variants(img_bgr: np.ndarray, scale: int, variant_id: int) -> np.ndarray:
        """
        Why these steps:
          - scale up: small UI digits OCR better
          - adaptive threshold: robust to UI glow/gradient backgrounds
          - morphology open/close: clean coin icon / noise
          - denoise: stabilize strokes
        """
        import cv2  # type: ignore

        gray = OCRReader._to_gray(img_bgr)
        if scale > 1:
            gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        if variant_id == 0:
            return OCRReader._gold_mask_variant(img_bgr, scale)

        if variant_id == 1:
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
            k = np.ones((2, 2), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
            return cv2.fastNlMeansDenoising(thr, None, 18, 7, 21)

        if variant_id == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            eq = clahe.apply(gray)
            eq = cv2.GaussianBlur(eq, (3, 3), 0)
            _, thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k = np.ones((2, 2), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
            return thr

        if variant_id == 3:
            bgr = img_bgr
            if scale > 1:
                bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # узкий диапазон под “желто-оранжевые” цифры
            lower = np.array([8, 80, 80])
            upper = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # подчистить шум и подчеркнуть разрывы дуг у «3», чтобы не превращались в «8»
            k = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.erode(mask, k, iterations=1)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)

            # делаем “черный текст на белом”, OCR так стабильнее
            mask = cv2.bitwise_not(mask)
            return mask

        # default fallback
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = np.ones((2, 2), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
        return cv2.fastNlMeansDenoising(thr, None, 12, 7, 21)

    def _easyocr_read(self, img: np.ndarray) -> OCRResult:
        if not self._easy_reader:
            return OCRResult(None, 0.0, "", "easyocr")

        # easyocr expects RGB
        rgb = img
        if len(img.shape) == 2:
            rgb = np.stack([img, img, img], axis=-1)
        else:
            rgb = img[..., ::-1]  # BGR->RGB

        try:
            results = self._easy_reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                allowlist="0123456789",
                decoder="beamsearch",
                beamWidth=10,
                min_size=8,
                mag_ratio=1.0,
                canvas_size=2560,
                contrast_ths=0.2,
                adjust_contrast=0.8,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.3,
            )
        except Exception:
            return OCRResult(None, 0.0, "", "easyocr")

        best_val: Optional[int] = None
        best_conf = 0.0
        best_raw = ""
        for _bbox, text, conf in results:
            val = parse_price_to_int(text)
            if val is not None and float(conf) >= best_conf:
                best_val = val
                best_conf = float(conf)
                best_raw = text

        return OCRResult(best_val, best_conf, best_raw, "easyocr")

    def _tesseract_read(self, img: np.ndarray) -> OCRResult:
        if not self._tesseract_ok:
            return OCRResult(None, 0.0, "", "tesseract")
        import pytesseract  # type: ignore

        # digits and separators only
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789 ,."
        try:
            text = pytesseract.image_to_string(img, config=cfg)
        except Exception:
            return OCRResult(None, 0.0, "", "tesseract")

        val = parse_price_to_int(text)
        # tesseract confidence is not easily available via image_to_string; use heuristic
        conf = 0.55 if val is not None else 0.0
        return OCRResult(val, conf, text.strip(), "tesseract")

    def read_price(self, crop_bgr: np.ndarray, timeout_s: float, variants: int, scales: Sequence[int]) -> OCRResult:
        start = time.time()
        candidates: list[OCRResult] = []

        for scale in scales:
            for v in range(max(1, variants)):
                if time.time() - start > timeout_s:
                    break
                pre = self.preprocess_variants(crop_bgr, scale=scale, variant_id=v)
                if self.use_easyocr:
                    candidates.append(self._easyocr_read(pre))
                if self.use_tesseract:
                    candidates.append(self._tesseract_read(pre))

        # pick best by (confidence, has value)
        candidates.sort(key=lambda r: (r.value is not None, r.confidence), reverse=True)

        best = candidates[0] if candidates else OCRResult(None, 0.0, "", "none")
        # Majority vote if multiple same values appear (helps when confidence is noisy)
        if len(candidates) >= 3:
            freq: dict[int, int] = {}
            for c in candidates:
                if c.value is not None:
                    freq[c.value] = freq.get(c.value, 0) + 1
            if freq:
                top_val = max(freq.items(), key=lambda kv: kv[1])[0]
                # keep best candidate with that value if close enough
                best_same = next((c for c in candidates if c.value == top_val), None)
                if best_same and (best_same.confidence >= best.confidence * 0.85):
                    best = best_same

        if best.value is None or best.confidence < self.min_conf:
            return OCRResult(best.value, best.confidence, best.raw_text, best.engine)

        return best


def safe_crop(img_bgr: np.ndarray, rect: Rect) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1 = max(0, rect.x)
    y1 = max(0, rect.y)
    x2 = min(w, rect.x + rect.w)
    y2 = min(h, rect.y + rect.h)
    if x2 <= x1 or y2 <= y1:
        return img_bgr[0:1, 0:1].copy()
    return img_bgr[y1:y2, x1:x2].copy()
