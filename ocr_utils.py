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
        if crop_bgr is None or crop_bgr.size == 0:
            return OCRResult(None, 0.0, "", "error_empty_crop")

        start = time.time()
        candidates: list[OCRResult] = []

        # ОПТИМИЗАЦИЯ: только scale=3, variants=0,1 (как у вас было)
        try:
            for scale in [3]:
                for v in [0, 1]:
                    if time.time() - start > timeout_s:
                        break
                    pre = self.preprocess_variants(crop_bgr, scale=scale, variant_id=v)
                    if self.use_easyocr:
                        result = self._easyocr_read(pre)
                        # РАННИЙ ВЫХОД при хорошем результате
                        if result.value is not None and result.confidence > 0.7:
                            # Санитарная проверка перед ранним выходом
                            if self._is_price_reasonable(result.value):
                                return result
                        candidates.append(result)
                    if self.use_tesseract:
                        candidates.append(self._tesseract_read(pre))
        except Exception as e:
            logging.getLogger(__name__).error("OCR processing error: %s", e)
            return OCRResult(None, 0.0, "", f"error_{type(e).__name__}")

        # pick best by (confidence, has value)
        candidates.sort(key=lambda r: (r.value is not None, r.confidence), reverse=True)

        best = candidates[0] if candidates else OCRResult(None, 0.0, "", "none")

        # Majority vote if multiple same values appear
        if len(candidates) >= 3:
            freq: dict[int, int] = {}
            for c in candidates:
                if c.value is not None:
                    freq[c.value] = freq.get(c.value, 0) + 1
            if freq:
                top_val = max(freq.items(), key=lambda kv: kv[1])[0]
                best_same = next((c for c in candidates if c.value == top_val), None)
                if best_same and (best_same.confidence >= best.confidence * 0.85):
                    best = best_same

        # НОВОЕ: санитарные проверки
        if best.value is not None:
            best = self._sanitize_3_8_confusion(best, candidates)
            best = self._sanitize_missing_zeros(best, candidates)

        return best

    def _sanitize_missing_zeros(self, best: OCRResult, all_candidates: list[OCRResult]) -> OCRResult:
        """
        Исправляет пропущенные нули: 300 -> 3000
        Логика:
        1. Если цена подозрительно маленькая (< 1000) и confidence не идеальна
        2. Проверяем, есть ли среди кандидатов варианты с дополнительным нулём
        3. Также проверяем контекст: если соседние цены ~2800-3000, то 300 явно ошибка
        """
        if best.value is None or best.value >= 1000:
            return best

        # Если confidence идеальна (1.0) - доверяем
        if best.confidence >= 0.98:
            return best

        price_str = str(best.value)

        # Проверяем варианты с дополнительными нулями
        # 300 -> [3000, 30000]
        possible_fixes = []

        # Вариант 1: добавить 0 в конец
        try:
            fix1 = int(price_str + '0')
            possible_fixes.append(fix1)
        except ValueError:
            pass

        # Вариант 2: добавить 00 в конец (на случай 30 -> 3000)
        if best.value < 100:
            try:
                fix2 = int(price_str + '00')
                possible_fixes.append(fix2)
            except ValueError:
                pass

        # Ищем эти варианты среди кандидатов
        best_fix = None
        best_fix_conf = 0.0

        for candidate in all_candidates:
            if candidate.value in possible_fixes:
                if candidate.confidence > best_fix_conf:
                    best_fix = candidate
                    best_fix_conf = candidate.confidence

        # Если нашли вариант с приемлемой confidence - используем
        if best_fix and best_fix_conf >= best.confidence * 0.7:
            logging.getLogger(__name__).info(
                "Missing zeros fix: %d->%d (conf %.2f->%.2f)",
                best.value, best_fix.value, best.confidence, best_fix_conf
            )
            return best_fix

        # Даже если нет кандидата, но цена явно неправильная - принудительно исправляем
        # Эвристика: если цена < 500 и confidence < 0.9 - скорее всего пропущен ноль
        if best.value < 500 and best.confidence < 0.90 and possible_fixes:
            fixed_value = possible_fixes[0]  # Берем первый вариант (с одним нулём)
            logging.getLogger(__name__).info(
                "Missing zeros fix (forced): %d->%d (conf %.2f, suspicious low price)",
                best.value, fixed_value, best.confidence
            )
            return OCRResult(
                value=fixed_value,
                confidence=best.confidence * 0.85,
                raw_text=best.raw_text + '0',
                engine=best.engine
            )

        return best

    @staticmethod
    def _is_price_reasonable(price: int) -> bool:
        """Проверка, что цена в разумных пределах"""
        return 1 <= price <= 999_999_999

    @staticmethod
    def _gold_mask_variant_v2(img_bgr: np.ndarray, scale: int) -> np.ndarray:
        """
        Улучшенная версия gold mask для лучшего распознавания "3"
        Ключевые изменения:
        - Более широкий диапазон HSV для захвата всех оттенков
        - МИНИМАЛЬНАЯ морфология (только 1 итерация close)
        - Дополнительная проверка на сохранение разрывов в "3"
        """
        import cv2
        import numpy as np

        bgr = img_bgr
        if scale > 1:
            bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                             interpolation=cv2.INTER_CUBIC)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Расширенный диапазон для золотых цифр
        lower = np.array([8, 60, 60], dtype=np.uint8)  # Было [10, 70, 70]
        upper = np.array([45, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # КРИТИЧНО: только 1 итерация close с минимальным ядром
        k = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # НЕ делаем erosion - он ломает "3"

        return mask

    def _sanitize_3_8_confusion(self, best: OCRResult, all_candidates: list[OCRResult]) -> OCRResult:
        """
        Исправляет типичную ошибку: 3xxx распознается как 8xxx
        Логика:
        1. Если лучший результат начинается с 8, проверяем есть ли вариант с 3
        2. Если есть вариант с 3 (замена первой цифры 8->3), и его confidence близка - выбираем его
        3. Используем дополнительную проверку через частоту цифр в сыром тексте
        """
        if best.value is None or best.value < 8000:
            return best

        # Проверяем, начинается ли с 8
        price_str = str(best.value)
        if not price_str.startswith('8'):
            return best

        # Создаем альтернативный вариант с 3
        alt_price_str = '3' + price_str[1:]
        try:
            alt_price = int(alt_price_str)
        except ValueError:
            return best

        # Ищем среди кандидатов вариант с этой ценой
        alt_candidate = None
        for c in all_candidates:
            if c.value == alt_price:
                alt_candidate = c
                break

        # КЛЮЧЕВАЯ ЭВРИСТИКА: анализ сырого текста
        # Считаем, сколько раз встречается "3" vs "8" в raw_text всех кандидатов
        count_3 = sum(c.raw_text.count('3') for c in all_candidates if c.raw_text)
        count_8 = sum(c.raw_text.count('8') for c in all_candidates if c.raw_text)

        # Если в сыром тексте "3" встречается чаще или наравне с "8",
        # и разница в confidence небольшая - берем вариант с 3
        if count_3 >= count_8:
            if alt_candidate and alt_candidate.confidence >= best.confidence * 0.75:
                logging.getLogger(__name__).info(
                    "3/8 confusion fix: %d->%d (conf %.2f->%.2f, raw_count 3:%d 8:%d)",
                    best.value, alt_price, best.confidence, alt_candidate.confidence,
                    count_3, count_8
                )
                return alt_candidate

            # Даже если нет точного кандидата, но count_3 значительно больше - принудительная замена
            if count_3 > count_8 * 1.5 and best.confidence < 0.95:
                logging.getLogger(__name__).info(
                    "3/8 confusion fix (forced): %d->%d (conf %.2f, raw_count 3:%d 8:%d)",
                    best.value, alt_price, best.confidence, count_3, count_8
                )
                return OCRResult(
                    value=alt_price,
                    confidence=best.confidence * 0.9,  # Немного снижаем уверенность
                    raw_text=best.raw_text.replace('8', '3', 1),
                    engine=best.engine
                )

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
