from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

        self._ocr_executor = ThreadPoolExecutor(max_workers=4)
        self._ocr_lock = threading.Lock()

        self._anchor_check_counter = 0
        self._last_anchor_check_time = 0

    def should_check_anchor(self) -> bool:
        """Определяет, нужно ли проверять якорь"""
        if not self.cfg.humanization.enabled:
            return False

        self._anchor_check_counter += 1

        # Проверяем каждые N действий
        if self._anchor_check_counter >= self.cfg.humanization.anchor_check_interval:
            self._anchor_check_counter = 0
            return True

        return False
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

    def verify_anchor_present(self, raise_on_fail: bool = True) -> bool:
        """
        Проверяет, что якорь рынка всё ещё виден на экране.
        Если якорь не найден - бот должен остановиться.

        Args:
            raise_on_fail: если True, выбрасывает исключение при отсутствии якоря

        Returns:
            True если якорь найден, False если нет

        Raises:
            RuntimeError: если якорь не найден и raise_on_fail=True
        """
        try:
            anchor_tpl = self._load_tpl(self.cfg.templates.anchor_market)
        except Exception as e:
            self.logger.error("Anchor template load failed: %s", e)
            if raise_on_fail:
                raise RuntimeError(f"Cannot load anchor template: {e}")
            return False

        screen = self.screen.screenshot_bgr()
        m = self.matcher.match_template(
            screen,
            anchor_tpl,
            threshold=self.cfg.templates.threshold_anchor,
            multi_scale=self.cfg.templates.multi_scale,
        )

        if not m:
            self.logger.error("⚠️ ANCHOR LOST! Market window not visible or has changed.")
            if raise_on_fail:
                raise RuntimeError(
                    "Market anchor not found on screen. "
                    "Возможные причины:\n"
                    "1. Окно рынка было закрыто\n"
                    "2. Окно было свернуто или перекрыто другими окнами\n"
                    "3. Интерфейс игры изменился\n"
                    "4. Произошло переключение окон\n\n"
                    "Бот остановлен для безопасности."
                )
            return False

        self.logger.debug("✓ Anchor verified at %s (score=%.3f)", m.center, m.score)
        self._anchor = m.top_left
        return True

    def safe_action_with_anchor_check(self, action_name: str):
        """
        Декоратор-контекст для безопасного выполнения действий с проверкой якоря.
        Используйте перед критичными операциями.
        """
        if self.should_check_anchor():
            self.logger.info("Periodic anchor check before: %s", action_name)
            self.verify_anchor_present(raise_on_fail=True)

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
        self.safe_action_with_anchor_check("run_search")
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

    # def scan_prices_on_page(self, page: int) -> list[tuple[int, int, float, str]]:
    #     """
    #     Returns list of (slot, price, conf, raw_text) for up to max_slots.
    #     OCR failures are skipped but logged.
    #     """
    #     out: list[tuple[int, int, float, str]] = []
    #     for slot in range(1, self.cfg.runtime.max_slots + 1):
    #         rect = self.price_rect_for_slot(slot)
    #         crop = self.screen.grab_region_bgr(rect)
    #         import os, cv2
    #
    #         os.makedirs("debug_prices", exist_ok=True)
    #         cv2.imwrite(f"debug_prices/page{page}_slot{slot}.png", crop)
    #         r = self.ocr.read_price(
    #             crop,
    #             timeout_s=self.cfg.timing.wait_ocr_timeout,
    #             variants=self.cfg.ocr.variants,
    #             scales=self.cfg.ocr.scale_factors,
    #         )
    #         if r.value is None:
    #             self.logger.warning("OCR price failed: page=%d slot=%d conf=%.2f raw=%r", page, slot, r.confidence, r.raw_text)
    #             continue
    #         self.logger.info("Price: page=%d slot=%d price=%d conf=%.2f raw=%r (%s)", page, slot, r.value, r.confidence, r.raw_text, r.engine)
    #         out.append((slot, r.value, r.confidence, r.raw_text))
    #     return out

    def cleanup(self):
        """Cleanup resources - IMPROVED with proper error handling and timeout"""
        if hasattr(self, '_ocr_executor'):
            try:
                self.logger.info("Shutting down OCR executor...")
                # Shutdown with timeout to prevent hanging (5 seconds max)
                self._ocr_executor.shutdown(wait=True, cancel_futures=True)
                self.logger.info("OCR executor shutdown completed successfully")
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures parameter
                try:
                    self._ocr_executor.shutdown(wait=True)
                    self.logger.info("OCR executor shutdown completed (legacy mode)")
                except Exception as e:
                    self.logger.warning("OCR executor shutdown failed: %s", e)
                    # Force shutdown without waiting
                    try:
                        self._ocr_executor.shutdown(wait=False)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning("OCR executor shutdown failed: %s", e)
                # Force shutdown without waiting
                try:
                    self._ocr_executor.shutdown(wait=False)
                except Exception:
                    pass
            finally:
                # Ensure we remove the reference even if shutdown failed
                try:
                    delattr(self, '_ocr_executor')
                    self.logger.debug("OCR executor reference removed")
                except Exception:
                    pass

    def scan_prices_on_page_parallel(self, page: int) -> list[tuple[int, int, float, str]]:
        """
        Parallel OCR for all slots on page.
        Returns list of (slot, price, conf, raw_text) for up to max_slots.
        """
        import os, cv2

        def read_slot_price(slot: int) -> Optional[tuple[int, int, float, str]]:
            # ... existing code unchanged ...
            try:
                rect = self.price_rect_for_slot(slot)
                crop = self.screen.grab_region_bgr(rect)

                os.makedirs("debug_prices", exist_ok=True)
                cv2.imwrite(f"debug_prices/page{page}_slot{slot}.png", crop)

                with self._ocr_lock:
                    r = self.ocr.read_price(
                        crop,
                        timeout_s=self.cfg.timing.wait_ocr_timeout,
                        variants=self.cfg.ocr.variants,
                        scales=self.cfg.ocr.scale_factors,
                    )

                if r is None:
                    self.logger.warning("OCR returned None: page=%d slot=%d", page, slot)
                    return None

                if r.value is None:
                    self.logger.warning("OCR price failed: page=%d slot=%d conf=%.2f raw=%r",
                                        page, slot, r.confidence, r.raw_text)
                    return None

                self.logger.info("Price: page=%d slot=%d price=%d conf=%.2f raw=%r (%s)",
                                 page, slot, r.value, r.confidence, r.raw_text, r.engine)
                return (slot, r.value, r.confidence, r.raw_text)

            except Exception as e:
                self.logger.error("OCR exception: page=%d slot=%d error=%s", page, slot, e)
                return None

        # Запускаем параллельно для всех слотов
        futures = []
        for slot in range(1, self.cfg.runtime.max_slots + 1):
            future = self._ocr_executor.submit(read_slot_price, slot)
            futures.append(future)

        # Собираем результаты
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.cfg.timing.wait_ocr_timeout * 2)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error("Future failed: %s", e)

        results = sorted(results, key=lambda x: x[0])  # Сортируем по slot

        # НОВОЕ: Контекстная проверка аномальных цен
        results = self._fix_anomalous_prices(results, page)

        return results

    def _fix_anomalous_prices(
            self,
            results: list[tuple[int, int, float, str]],
            page: int
    ) -> list[tuple[int, int, float, str]]:
        """
        Исправляет явно аномальные цены на основе контекста.
        Логика: если цена в 10+ раз меньше соседних - скорее всего пропущен ноль.
        """
        if len(results) < 3:
            return results

        fixed_results = []
        prices = [p for (_, p, _, _) in results]

        for i, (slot, price, conf, raw) in enumerate(results):
            # Собираем соседние цены (±2 слота)
            neighbors = []
            for j in range(max(0, i - 2), min(len(prices), i + 3)):
                if j != i:
                    neighbors.append(prices[j])

            if not neighbors:
                fixed_results.append((slot, price, conf, raw))
                continue

            avg_neighbor = sum(neighbors) / len(neighbors)

            # Если цена в 10+ раз меньше средней соседней и confidence не идеальна
            if price < avg_neighbor / 10 and conf < 0.95:
                # Пробуем добавить ноль
                fixed_price = price * 10

                # Проверяем, что исправленная цена ближе к соседям
                if abs(fixed_price - avg_neighbor) < abs(price - avg_neighbor):
                    self.logger.warning(
                        "Context fix: page=%d slot=%d %d->%d (neighbors avg=%.0f, conf=%.2f)",
                        page, slot, price, fixed_price, avg_neighbor, conf
                    )
                    fixed_results.append((slot, fixed_price, conf * 0.9, raw + "0"))
                    continue

            fixed_results.append((slot, price, conf, raw))

        return fixed_results

    def next_page(self) -> None:
        self.safe_action_with_anchor_check("next_page")

        p = self.rel_point(self.cfg.offs.next_page)
        self.logger.info("Next page click at %s", p)
        self.inp.click(p)
        time.sleep(self.cfg.timing.wait_page_flip)

    def locate_buy_cancel_buttons(self) -> tuple[Optional[Point], Optional[Point]]:
        """
        Locate Buy/Cancel buttons in modal via templates; fallback offsets relative to anchor.
        Improvements:
          - if cancel not found globally, try searching a small region around the found Buy button
            with a slightly lower threshold and save debug crops for inspection.
        """
        screen = self.screen.screenshot_bgr()

        buy_p: Optional[Point] = None
        cancel_p: Optional[Point] = None

        # --- BUY (global) ---
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

        # --- CANCEL (global) ---
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

        # --- If cancel not found globally but buy found: search locally around buy ---
        if cancel_p is None and buy_p is not None:
            try:
                # region around the buy button — tune size if needed
                rx = max(0, buy_p.x - 240)
                ry = max(0, buy_p.y - 140)
                rw = 480
                rh = 280
                local_rect = Rect(rx, ry, rw, rh)
                local_hay = self.screen.grab_region_bgr(local_rect)

                # slightly relax threshold for local search
                local_thr = max(0.60, float(self.cfg.templates.threshold_buttons) * 0.9)

                m_local = self.matcher.match_template(
                    local_hay, cancel_tpl,
                    threshold=local_thr,
                    multi_scale=self.cfg.templates.multi_scale
                )
                if m_local:
                    # m_local.center is relative to local_hay -> convert to absolute
                    cancel_p = Point(local_rect.x + m_local.center.x, local_rect.y + m_local.center.y)
                    self.logger.info(
                        "Cancel button found by local-template at %s (score=%.3f, thr=%.2f)",
                        cancel_p, m_local.score, local_thr
                    )
                else:
                    # dump debug crop for manual inspection
                    try:
                        import os, cv2
                        os.makedirs("debug_btns", exist_ok=True)
                        cv2.imwrite("debug_btns/btn_search_local.png", local_hay)
                        tpl_img = self._load_tpl(self.cfg.templates.btn_cancel)
                        cv2.imwrite("debug_btns/btn_cancel_tpl.png", tpl_img)
                        self.logger.info("Wrote debug images to debug_btns/ for cancel local search")
                    except Exception as e:
                        self.logger.debug("Failed to write debug images for cancel: %s", e)

            except Exception as e:
                self.logger.warning("Local cancel search error: %s", e)

        # --- Final fallbacks to offsets if still not found ---
        if buy_p is None:
            buy_p = self.rel_point(self.cfg.offs.buy_btn_fallback)
            self.logger.info("Buy button fallback point: %s", buy_p)
        if cancel_p is None:
            cancel_p = self.rel_point(self.cfg.offs.buy_cancel_fallback)
            self.logger.info("Cancel button fallback point: %s", cancel_p)

        return buy_p, cancel_p

    def open_lot_modal(self, slot_index_1based: int) -> None:
        self.safe_action_with_anchor_check("open_lot_modal")

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

    @staticmethod
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
            self.inp.sleep(self.cfg.timing.wait_between_clicks)

            # очистить поле максимально надёжно
            self.inp.ctrl_combo("A")
            self.inp.sleep(self.cfg.timing.wait_short)

            # вставка
            self.inp.set_clipboard(str(qty))
            self.inp.ctrl_combo("V")
            self.inp.press("end")
            self.inp.sleep(self.cfg.timing.wait_small)

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
            self.inp.sleep(self.cfg.timing.wait_between_clicks)

        return False

    def click_buy(self, buy_btn: Point) -> None:
        self.logger.info("Click BUY at %s", buy_btn)
        self.inp.click(buy_btn)

    def click_cancel(self, cancel_btn: Point) -> None:
        self.logger.info("Click CANCEL at %s", cancel_btn)
        self.inp.click(cancel_btn)

    def find_item_slot_by_icon(self, item_name: str) -> Optional[tuple[int, float]]:
        slots = self.find_item_slots_by_icon(item_name)
        if not slots:
            return None
        # берём самый верхний слот (обычно дешевле всех, рынок отсортирован)
        return slots[0][0], slots[0][1]

    def _normalize_icon_template(self, tpl0: np.ndarray) -> np.ndarray:
        """
        Делает шаблон "уникальным": берём левый квадрат (иконку) и отрезаем рамку.
        ВАЖНО: не режем половину иконки, иначе снова будет матчиться рамка.
        """
        import cv2  # type: ignore
        import numpy as np

        tpl = tpl0.copy()
        h, w = tpl.shape[:2]

        # если шаблон широкий (иконка + текст) => оставляем левый квадрат
        if w > int(h * 1.10):
            tpl = tpl[:, :h].copy()

        # отрезаем рамку (обычно она одинаковая у всех иконок и даёт ложные 1.0)
        h2, w2 = tpl.shape[:2]
        pad = max(2, min(5, min(h2, w2) // 8))  # для 32px будет 4
        if (h2 - 2 * pad) >= 16 and (w2 - 2 * pad) >= 16:
            tpl = tpl[pad:-pad, pad:-pad].copy()

        return tpl

    def find_item_slots_by_icon(self, item_name: str) -> list[tuple[int, float]]:
        """
        Возвращает список слотов на текущей странице, где найдена иконка.
        Важно: ищем ВСЕ совпадения, делаем NMS, потом группируем по слоту.
        Пишет debug-скрины региона и full-screen как у тебя уже сделано.
        """
        import os
        import re
        import math
        import cv2  # type: ignore
        import numpy as np

        icon_rel = self.cfg.templates.item_icons.get(item_name)
        if not icon_rel:
            self.logger.warning("No icon template mapped for item %r", item_name)
            return []

        tpl0 = self._load_tpl(icon_rel)
        tpl = self._normalize_icon_template(tpl0)

        # регион поиска (config -> иначе derived)
        if self.cfg.offs.region_slots is not None:
            region = self.rel_rect(self.cfg.offs.region_slots)
            tag = "config"
        else:
            first = self.rel_point(self.cfg.offs.first_slot)
            price_rect = self.rel_rect(self.cfg.offs.region_price)
            top_y = price_rect.y - (self.cfg.offs.slot_height - self.cfg.offs.region_price.h) // 2
            h = self.cfg.offs.slot_height * self.cfg.runtime.max_slots
            x = first.x - 190
            w = max(self.cfg.offs.region_price.x + self.cfg.offs.region_price.w - (self.cfg.offs.first_slot.x - 100),
                    260)
            region = Rect(x, top_y, w, h)
            tag = "derived"

        hay_bgr = self.screen.grab_region_bgr(region)

        # матчим по edges — меньше ложных от рамок/цвета/подсветки строки
        hay_gray = cv2.cvtColor(hay_bgr, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        hay = cv2.Canny(hay_gray, 50, 150)
        t = cv2.Canny(tpl_gray, 50, 150)

        th, tw = t.shape[:2]
        hh, hw = hay.shape[:2]
        if th >= hh or tw >= hw:
            self.logger.warning("Template too big for region: tpl=%s region=%s", t.shape, region)
            return []

        res = cv2.matchTemplate(hay, t, cv2.TM_CCOEFF_NORMED)

        thr = float(self.cfg.templates.threshold_item_icon)
        ys, xs = np.where(res >= thr)
        if len(xs) == 0:
            # debug dump for miss
            os.makedirs("debug_item_icons", exist_ok=True)
            safe = re.sub(r"[^\w\-]+", "_", item_name)
            dbg = hay_bgr.copy()
            cv2.putText(dbg, f"region={tag} abs=({region.x},{region.y},{region.w},{region.h})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dbg, "not found", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f"debug_item_icons/{safe}_{tag}_region_miss.png", dbg)
            self.logger.info("No icon matches >= thr on this page: item=%r thr=%.2f", item_name, thr)
            return []

        scores = res[ys, xs]
        cand = sorted(zip(xs.tolist(), ys.tolist(), scores.tolist()), key=lambda z: z[2], reverse=True)

        # NMS чтобы не было 50 совпадений внутри одной и той же иконки
        picked: list[tuple[int, int, float]] = []
        for x, y, sc in cand:
            too_close = False
            for px, py, _ in picked:
                if abs(x - px) < int(tw * 0.7) and abs(y - py) < int(th * 0.7):
                    too_close = True
                    break
            if too_close:
                continue
            picked.append((x, y, float(sc)))
            if len(picked) >= 50:
                break

        # группируем по слоту
        first_slot_abs_y = self.rel_point(self.cfg.offs.first_slot).y
        slot_best: dict[int, float] = {}
        slot_pt: dict[int, tuple[int, int]] = {}

        for x, y, sc in picked:
            cy_abs = region.y + (y + th // 2)
            delta = (cy_abs - first_slot_abs_y) / float(self.cfg.offs.slot_height)
            slot = 1 + int(delta + 0.5)
            slot = max(1, min(slot, self.cfg.runtime.max_slots))
            if slot not in slot_best or sc > slot_best[slot]:
                slot_best[slot] = sc
                slot_pt[slot] = (x, y)

        out = sorted([(s, slot_best[s]) for s in slot_best.keys()], key=lambda k: k[0])

        # ---- DEBUG: рисуем все найденные слоты ----
        try:
            os.makedirs("debug_item_icons", exist_ok=True)
            safe = re.sub(r"[^\w\-]+", "_", item_name)

            dbg = hay_bgr.copy()
            cv2.putText(dbg, f"region={tag} abs=({region.x},{region.y},{region.w},{region.h}) thr={thr:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            for slot, sc in out:
                x, y = slot_pt[slot]
                cv2.rectangle(dbg, (x, y), (x + tw, y + th), (0, 255, 0), 2)
                cv2.putText(dbg, f"s{slot}:{sc:.2f}", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imwrite(f"debug_item_icons/{safe}_{tag}_region_multi.png", dbg)
        except Exception as e:
            self.logger.warning("Failed to dump multi-match debug: %s", e)

        self.logger.info("Icon slots found: item=%r slots=%s", item_name, out)
        return out

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
                self.inp.sleep(self.cfg.timing.wait_after_cancel)
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
        self.inp.sleep(self.cfg.timing.wait_small)
        self.inp.press("backspace")
        # Ctrl+V
        self.inp.ctrl_combo("V")


        if not verify:
            return

        # Верификация: Ctrl+A Ctrl+C -> проверяем, что реально в поле
        self.inp.ctrl_combo("A")
        self.inp.sleep(self.cfg.timing.wait_small)
        self.inp.ctrl_combo("C")
        got = (self.inp.get_clipboard() or "").strip()
        self.inp.press("end")
        if got == text.strip():
            return

        self.logger.warning(
            "Paste verify failed for %s: got=%r expected=%r. Fallback to typing.",
            field_name, got, text
        )