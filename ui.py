from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import numpy as np

from config import AppConfig
from input_controller import InputController
from ocr_utils import OCRReader
from models import Category, Point, Rect


@dataclass
class Match:
    top_left: Point
    center: Point
    score: float



class TemplateMatcher:
    def __init__(self, logger: object) -> None:
        self.logger = logger

    @staticmethod
    def _load_template(path: Path) -> np.ndarray:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Template not found or unreadable: {path}")
        return img

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return img
        return np.stack([img, img, img], axis=-1)

    def match_template(
        self,
        haystack_bgr: np.ndarray,
        template_bgr: np.ndarray,
        threshold: float,
        multi_scale: bool,
        scales: Optional[list[float]] = None,
    ) -> Optional[Match]:
        """
        OpenCV matchTemplate is fast for small UI icons/buttons.
        Multi-scale helps when UI scale differs (windowed/fullscreen, DPI settings).
        """
        import cv2  # type: ignore

        hay = self._to_bgr(haystack_bgr)
        tpl = self._to_bgr(template_bgr)

        if scales is None:
            scales = [1.0, 0.95, 1.05, 0.9, 1.1] if multi_scale else [1.0]

        best: Optional[Match] = None
        for s in scales:
            if s != 1.0:
                tpl_s = cv2.resize(tpl, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            else:
                tpl_s = tpl

            th, tw = tpl_s.shape[:2]
            hh, hw = hay.shape[:2]
            if th >= hh or tw >= hw:
                continue

            res = cv2.matchTemplate(hay, tpl_s, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                top_left = Point(int(max_loc[0]), int(max_loc[1]))
                center = Point(int(max_loc[0] + tw // 2), int(max_loc[1] + th // 2))
                m = Match(top_left=top_left, center=center, score=float(max_val))
                if best is None or m.score > best.score:
                    best = m

        return best


class Screen:
    def __init__(self, logger: object) -> None:
        self.logger = logger

    def screenshot_bgr(self) -> np.ndarray:
        """
        Uses pyautogui.screenshot -> PIL -> numpy.
        """
        import pyautogui  # type: ignore
        import cv2  # type: ignore

        pil = pyautogui.screenshot()
        img = np.array(pil)  # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def grab_region_bgr(self, rect: Rect) -> np.ndarray:
        import pyautogui  # type: ignore
        import cv2  # type: ignore

        pil = pyautogui.screenshot(region=(rect.x, rect.y, rect.w, rect.h))
        img = np.array(pil)  # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class MarketUI:
    def __init__(self, cfg: AppConfig, inp: InputController, ocr: OCRReader, logger: object) -> None:
        self.cfg = cfg
        self.inp = inp
        self.ocr = ocr
        self.logger = logger

        self.screen = Screen(logger)
        self.matcher = TemplateMatcher(logger)

        self._tpl_cache: dict[str, np.ndarray] = {}
        self._anchor: Optional[Point] = None

        # cache for item icon y within current page (still re-validated before buying)
        self._item_icon_cache: dict[str, int] = {}
        self._last_search_query: Optional[str] = None

    def _tpl_path(self, rel: str) -> Path:
        return (self.cfg.templates.templates_dir / rel).resolve()

    def _load_tpl(self, rel: str) -> np.ndarray:
        if rel in self._tpl_cache:
            return self._tpl_cache[rel]
        tpl = self.matcher._load_template(self._tpl_path(rel))
        self._tpl_cache[rel] = tpl
        return tpl

    def find_anchor(self) -> Optional[Point]:
        """
        Finds market anchor on the screen via template matching.
        All coordinates are then computed relative to it.
        """
        try:
            anchor_tpl = self._load_tpl(self.cfg.templates.anchor_market)
        except Exception as e:
            self.logger.error("Anchor template load failed: %s", e)
            return None

        screen = self.screen.screenshot_bgr()
        m = self.matcher.match_template(
            screen,
            anchor_tpl,
            threshold=self.cfg.templates.threshold_anchor,
            multi_scale=self.cfg.templates.multi_scale,
        )
        if not m:
            self.logger.error("Anchor not found (threshold=%.2f).", self.cfg.templates.threshold_anchor)
            return None

        self._anchor = m.top_left
        self.logger.info("Anchor found. top_left=%s center=%s score=%.3f", m.top_left, m.center, m.score)
        return m.center

    def anchor(self) -> Point:
        if self._anchor is None:
            a = self.find_anchor()
            if a is None:
                raise RuntimeError("Cannot continue: market anchor not found.")
        return self._anchor  # type: ignore[return-value]

    def rel_point(self, off: Point) -> Point:
        a = self.anchor()
        return Point(a.x + off.x, a.y + off.y)

    def rel_rect(self, off_rect: Rect, dy: int = 0) -> Rect:
        a = self.anchor()
        return Rect(a.x + off_rect.x, a.y + off_rect.y + dy, off_rect.w, off_rect.h)

    def open_category(self, cat: Category) -> None:
        off = self.cfg.offs.cat_trophy if cat == Category.TROPHY else self.cfg.offs.cat_res
        p = self.rel_point(off)
        self.logger.info("Open category %s at %s (dbl click)", cat.value, p)
        self.inp.dblclick(p)

    def focus_search(self) -> None:
        p = self.rel_point(self.cfg.offs.anchor_search)
        self.logger.info("Focus search field at %s", p)
        self.inp.click(p)

    def run_search(self, query: str) -> None:
        self.focus_search()
        self.inp.sleep(0.1)

        self._fill_text_field(query, verify=True, field_name="search")
        self.inp.sleep(self.cfg.timing.wait_after_search)
        self._last_search_query = query

    def slot_click_point(self, slot_index_1based: int) -> Point:
        i = max(1, min(slot_index_1based, self.cfg.runtime.max_slots))
        base = self.rel_point(self.cfg.offs.first_slot)
        dy = (i - 1) * self.cfg.offs.slot_height
        return Point(base.x, base.y + dy)

    def price_rect_for_slot(self, slot_index_1based: int) -> Rect:
        dy = (slot_index_1based - 1) * self.cfg.offs.slot_height
        return self.rel_rect(self.cfg.offs.region_price, dy=dy)

    def scan_prices_on_page(self, page: int) -> list[tuple[int, int, float, str]]:
        """
        Returns list of (slot, price, conf, raw_text) for up to max_slots.
        OCR failures are skipped but logged.
        """
        out: list[tuple[int, int, float, str]] = []
        for slot in range(1, self.cfg.runtime.max_slots + 1):
            rect = self.price_rect_for_slot(slot)
            crop = self.screen.grab_region_bgr(rect)
            import os, cv2

            os.makedirs("debug_prices", exist_ok=True)
            cv2.imwrite(f"debug_prices/page{page}_slot{slot}.png", crop)
            r = self.ocr.read_price(
                crop,
                timeout_s=self.cfg.timing.wait_ocr_timeout,
                variants=self.cfg.ocr.variants,
                scales=self.cfg.ocr.scale_factors,
            )
            if r.value is None:
                self.logger.warning("OCR price failed: page=%d slot=%d conf=%.2f raw=%r", page, slot, r.confidence, r.raw_text)
                continue
            self.logger.info("Price: page=%d slot=%d price=%d conf=%.2f raw=%r (%s)", page, slot, r.value, r.confidence, r.raw_text, r.engine)
            out.append((slot, r.value, r.confidence, r.raw_text))
        return out

    def next_page(self) -> None:
        p = self.rel_point(self.cfg.offs.next_page)
        self.logger.info("Next page click at %s", p)
        self.inp.click(p)
        time.sleep(self.cfg.timing.wait_page_flip)


    def locate_buy_cancel_buttons(self) -> tuple[Optional[Point], Optional[Point]]:
        """
        Locate Buy/Cancel buttons in modal via templates; fallback offsets relative to anchor.
        """
        screen = self.screen.screenshot_bgr()

        buy_p: Optional[Point] = None
        cancel_p: Optional[Point] = None

        try:
            buy_tpl = self._load_tpl(self.cfg.templates.btn_buy)
            m = self.matcher.match_template(
                screen, buy_tpl,
                threshold=self.cfg.templates.threshold_buttons,
                multi_scale=self.cfg.templates.multi_scale
            )
            if m:
                buy_p = m.center
                self.logger.info("Buy button found by template at %s (score=%.3f)", buy_p, m.score)
        except Exception as e:
            self.logger.warning("Buy button template error: %s", e)

        try:
            cancel_tpl = self._load_tpl(self.cfg.templates.btn_cancel)
            m = self.matcher.match_template(
                screen, cancel_tpl,
                threshold=self.cfg.templates.threshold_buttons,
                multi_scale=self.cfg.templates.multi_scale
            )
            if m:
                cancel_p = m.center
                self.logger.info("Cancel button found by template at %s (score=%.3f)", cancel_p, m.score)
        except Exception as e:
            self.logger.warning("Cancel button template error: %s", e)

        if buy_p is None:
            buy_p = self.rel_point(self.cfg.offs.buy_btn_fallback)
            self.logger.info("Buy button fallback point: %s", buy_p)
        if cancel_p is None:
            cancel_p = self.rel_point(self.cfg.offs.buy_cancel_fallback)
            self.logger.info("Cancel button fallback point: %s", cancel_p)

        return buy_p, cancel_p

    def open_lot_modal(self, slot_index_1based: int) -> None:
        p = self.slot_click_point(slot_index_1based)
        self.logger.info("Open lot modal: slot=%d at %s (dbl click)", slot_index_1based, p)
        self.inp.dblclick(p)
        self.inp.sleep(self.cfg.timing.wait_open_modal)

    def read_seller_qty_from_modal(self, buy_btn: Point) -> Optional[int]:
        """
        Preferred method: click input field (offset from Buy), Ctrl+A, Ctrl+C, read clipboard.
        Fallback: OCR a small region around the input.
        """
        import pyperclip  # type: ignore

        input_p = Point(buy_btn.x + self.cfg.offs.buy_input_offset.x, buy_btn.y + self.cfg.offs.buy_input_offset.y)
        self.logger.info("Read seller qty: click input at %s", input_p)
        self.inp.click(input_p)
        self.inp.press_hotkey("ctrl", "a")
        self.inp.press_hotkey("ctrl", "c")

        try:
            txt = self.inp.get_clipboard()
        except Exception:
            txt = ""

        val = None
        if isinstance(txt, str) and txt.strip():
            m = re.search(r"\d+", txt.replace("\u00A0", " "))
            if m:
                try:
                    val = int(m.group(0))
                except ValueError:
                    val = None

        if val is not None:
            self.logger.info("Seller qty from clipboard: %d (raw=%r)", val, txt)
            return val

        # Fallback OCR: capture a small region around the input point
        # (best-effort; you may tune in config if needed)
        rect = Rect(input_p.x - 60, input_p.y - 20, 120, 40)
        crop = self.screen.grab_region_bgr(rect)
        r = self.ocr.read_price(
            crop,
            timeout_s=self.cfg.timing.wait_ocr_timeout,
            variants=self.cfg.ocr.variants,
            scales=self.cfg.ocr.scale_factors,
        )
        self.logger.info("Seller qty OCR fallback: %s conf=%.2f raw=%r", r.value, r.confidence, r.raw_text)
        return r.value

    def _parse_int_from_text(s: str) -> int:
        m = re.findall(r"\d+", (s or "").replace("\u00A0", " "))
        if not m:
            return 0
        return int("".join(m))

    def set_buy_quantity(self, buy_btn: Point, qty: int) -> bool:
        input_p = Point(
            buy_btn.x + self.cfg.offs.buy_input_offset.x,
            buy_btn.y + self.cfg.offs.buy_input_offset.y
        )

        for attempt in range(1, 4):
            self.inp.click(input_p)
            self.inp.sleep(0.12)

            # очистить поле максимально надёжно
            self.inp.ctrl_combo("A")
            self.inp.sleep(0.03)
            self.inp.press("backspace")
            self.inp.sleep(0.03)

            # вставка
            self.inp.set_clipboard(str(qty))
            self.inp.ctrl_combo("V")
            self.inp.press("end")
            self.inp.sleep(0.10)

            # verify: Ctrl+A Ctrl+C -> clipboard
            self.inp.ctrl_combo("A")
            self.inp.ctrl_combo("C")
            txt = self.inp.get_clipboard()
            got = self._parse_int_from_text(txt)

            if got == qty:
                return True

            self.logger.warning(
                "Paste verify failed for buy_qty: got=%r parsed=%d expected=%d (attempt %d/3)",
                txt, got, qty, attempt
            )
            self.inp.sleep(0.12)

        return False

    def click_buy(self, buy_btn: Point) -> None:
        self.logger.info("Click BUY at %s", buy_btn)
        self.inp.click(buy_btn)

    def click_cancel(self, cancel_btn: Point) -> None:
        self.logger.info("Click CANCEL at %s", cancel_btn)
        self.inp.click(cancel_btn)

    def find_item_slot_by_icon(self, item_name: str) -> Optional[tuple[int, float]]:
        """
        Find the slot where the item icon appears (template matching).
        We scan the list region (8 slots) and compute slot index by Y.
        """
        import math
        import os
        import cv2  # type: ignore

        icon_rel = self.cfg.templates.item_icons.get(item_name)
        if not icon_rel:
            self.logger.warning("No icon template mapped for item %r", item_name)
            return None

        # region for scanning slots
        regions: list[tuple[str, Rect]] = []
        if self.cfg.offs.region_slots is not None:
            regions.append(("config", self.rel_rect(self.cfg.offs.region_slots)))

        # derived best-effort column around the item icons
        first = self.rel_point(self.cfg.offs.first_slot)
        price_rect = self.rel_rect(self.cfg.offs.region_price)
        top_y = price_rect.y - (self.cfg.offs.slot_height - self.cfg.offs.region_price.h) // 2
        h = self.cfg.offs.slot_height * self.cfg.runtime.max_slots
        # focus on the icon + name column instead of the whole list width
        x = first.x - 100
        w = max(self.cfg.offs.region_price.x + self.cfg.offs.region_price.w - (self.cfg.offs.first_slot.x - 100), 260)
        derived_region = Rect(x, top_y, w, h)
        regions.append(("derived", derived_region))

        try:
            tpl = self._load_tpl(icon_rel)
        except Exception as e:
            self.logger.error("Item icon template load failed (%r): %s", item_name, e)
            return None

        chosen_region: Optional[Rect] = None
        m: Optional[Match] = None

        for source, region in regions:
            hay = self.screen.grab_region_bgr(region)
            m = self.matcher.match_template(
                hay,
                tpl,
                threshold=self.cfg.templates.threshold_item_icon,
                multi_scale=self.cfg.templates.multi_scale,
            )

            # Debug visualization of icon search
            try:
                os.makedirs("debug_item_icons", exist_ok=True)
                dbg = hay.copy()
                safe_name = re.sub(r"[^\w\-]+", "_", item_name)

                if m:
                    th, tw = tpl.shape[:2]
                    top_left = (m.top_left.x, m.top_left.y)
                    bottom_right = (m.top_left.x + tw, m.top_left.y + th)
                    cv2.rectangle(dbg, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.drawMarker(dbg, (m.center.x, m.center.y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                    suffix = f"slot{1 + int(math.floor((region.y + m.center.y - self.rel_point(self.cfg.offs.first_slot).y) / self.cfg.offs.slot_height))}_hit"
                else:
                    cv2.putText(dbg, "not found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    suffix = "miss"

                cv2.putText(dbg, f"region: {source}", (10, hay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite(f"debug_item_icons/{safe_name}_{source}_{suffix}.png", dbg)
            except Exception as e:  # pragma: no cover - debug only
                self.logger.warning("Failed to dump icon debug screenshot: %s", e)

            if m:
                chosen_region = region
                break

        if not m or chosen_region is None:
            self.logger.info("Item icon not found on current page: %r", item_name)
            return None

        # m.center is relative to region
        y_abs = chosen_region.y + m.center.y
        first_slot_abs_y = self.rel_point(self.cfg.offs.first_slot).y
        slot_idx = 1 + int(math.floor((y_abs - first_slot_abs_y) / self.cfg.offs.slot_height))
        slot_idx = max(1, min(slot_idx, self.cfg.runtime.max_slots))

        self.logger.info("Icon match: item=%r slot=%d score=%.3f", item_name, slot_idx, m.score)
        self._item_icon_cache[item_name] = slot_idx

        return slot_idx, m.score

    def close_modal_safely(self) -> None:
        """
        Закрывает модалку только если реально видна кнопка Cancel (по шаблону).
        Никаких fallback-кликов (чтобы не промахнуться, если модалка уже закрылась).
        """
        try:
            screen = self.screen.screenshot_bgr()
            cancel_tpl = self._load_tpl(self.cfg.templates.btn_cancel)
            m = self.matcher.match_template(
                screen, cancel_tpl,
                threshold=self.cfg.templates.threshold_buttons,
                multi_scale=self.cfg.templates.multi_scale
            )
            if m:
                self.logger.info("Close modal (cancel template) at %s (score=%.3f)", m.center, m.score)
                self.inp.click(m.center)
                self.inp.sleep(0.25)
        except Exception as e:
            self.logger.warning("close_modal_safely error: %s", e)

    def _fill_text_field(self, text: str, verify: bool = True, field_name: str = "field") -> None:
        """
        Пытаемся вставить как в старом боте (Ctrl+A / Ctrl+V),
        но если поле печатает 0x01/0x16 и текст не появился — падаем на typewrite.
        """
        self.inp.set_clipboard(text)

        # Ctrl+A (оставляем как ты просил)
        self.inp.ctrl_combo("A")

        # В старом v8 у тебя был backspace после Ctrl+A — оставим, это часто “фиксит” поле
        self.inp.press("backspace")

        # Ctrl+V
        self.inp.ctrl_combo("V")
        self.inp.sleep(0.12)

        if not verify:
            return

        # Верификация: Ctrl+A Ctrl+C -> проверяем, что реально в поле
        self.inp.ctrl_combo("A")
        self.inp.ctrl_combo("C")
        got = (self.inp.get_clipboard() or "").strip()
        self.inp.press("end")
        if got == text.strip():
            return

        self.logger.warning(
            "Paste verify failed for %s: got=%r expected=%r. Fallback to typing.",
            field_name, got, text
        )

        # Полная очистка и печать
        self.inp.ctrl_combo("A")
        self.inp.press("backspace", presses=80, interval=0.005)
        self.inp.type_text(text)
        self.inp.sleep(0.12)
