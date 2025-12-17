from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from config import AppConfig
from models import (
    Category,
    ItemMarketData,
    LotInfo,
    Mode,
    PurchaseResult,
    QuestDefinition,
    QuestItem,
)
from purchase_logger import PurchaseLogger
from ui import MarketUI
from planner import build_plan_one_page, execute_plan_one_page

def parse_quests_db(raw_list: list[dict]) -> list[QuestDefinition]:
    out: list[QuestDefinition] = []
    for row in raw_list:
        cat_s = str(row["cat"])
        cat = Category.TROPHY if cat_s == Category.TROPHY.value else Category.RESOURCES

        items_raw = row["item"]
        qty_raw = row["qty"]

        if isinstance(items_raw, list):
            items: list[QuestItem] = []
            assert isinstance(qty_raw, list), "qty must be list when item is list"
            for n, q in zip(items_raw, qty_raw):
                items.append(QuestItem(name=str(n), qty_per_set=int(q)))
        else:
            items = [QuestItem(name=str(items_raw), qty_per_set=int(qty_raw))]

        out.append(
            QuestDefinition(
                quest=str(row["quest"]),
                category=cat,
                reward60=int(row["r60"]),
                reward65=int(row["r65"]),
                items=items,
            )
        )
    return out


@dataclass
class BotContext:
    cfg: AppConfig
    ui: MarketUI
    logger: object
    mode: Mode
    level: int
    sets: int


class MarketBot:
    def __init__(self, ctx: BotContext) -> None:
        self.ctx = ctx
        default_path = Path("purchases_log.csv")
        self.purchase_logger = PurchaseLogger(default_path, append=True)

    def reward_for_level(self, q: QuestDefinition) -> int:
        return q.reward60 if self.ctx.level == 60 else q.reward65

    def run(self, quests: Iterable[QuestDefinition]) -> None:
        log = self.ctx.logger
        log.info("Bot started. level=%d sets=%d mode=%s", self.ctx.level, self.ctx.sets, self.ctx.mode.value)

        # find anchor once (will raise if missing)
        try:
            self.ctx.ui.anchor()
        except Exception as e:
            log.error("Cannot start: %s", e)
            return

        for q in quests:
            try:
                self.process_quest(q)
            except Exception as e:
                log.exception("Quest failed (skipping): %r error=%s", q.quest, e)

        log.info("Bot finished.")

    def process_quest(self, q: QuestDefinition) -> None:
        log = self.ctx.logger
        reward_per_set = self.reward_for_level(q)
        reward_total = reward_per_set * self.ctx.sets
        log.info("==== Quest: %s | reward_per_set=%d | reward_total=%d | cat=%s ====",
                 q.quest, reward_per_set, reward_total, q.category.value)

        # open category (may be simulated)
        if self.ctx.mode == Mode.RUN or self.ctx.cfg.runtime.simulate_ui_actions:
            self.ctx.ui.open_category(q.category)

        ui = self.ctx.ui
        cfg = self.ctx.cfg

        can_open_modals = (self.ctx.mode == Mode.RUN) or cfg.runtime.simulate_ui_actions
        do_buy = (self.ctx.mode == Mode.RUN)  # simulate_ui_actions => не жмем BUY

        items_data: list[ItemMarketData] = []
        plans_by_item: dict[str, object] = {}  # PlanResult, но без импорта типа (можно и типизировать)

        total_min_cost = 0

        # -------- Stage A: оценка выгодности --------
        for item in q.items:
            item2 = self._apply_image_mode_if_configured(item)
            required_qty = item2.qty_per_set * self.ctx.sets

            # --- image items: оставляем старую collect_item_data (там next_page допустим) ---
            if item2.use_image:
                budget_left = reward_total - total_min_cost  # важно: это безопасный ранний отсев
                data = self.collect_image_item_data(item2, required_qty, budget_left=budget_left)

                if data.min_cost is None:
                    log.info("Item data incomplete => skip quest. item=%r reason=%s", item2.name, data.reason)
                    return

                items_data.append(data)

                total_min_cost += data.min_cost
                continue

            # --- обычные items: строим план на 1 странице ---
            log.info("Collect/Plan item: %r required_qty=%d (one-page plan)", item2.name, required_qty)

            if self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions:
                ui.run_search(item2.name)

            # если мы не можем открывать модалки (pure simulate), делаем только грубую оценку p1*need
            if not can_open_modals:
                scanned = ui.scan_prices_on_page(page=1)
                p1 = next((p for (slot, p, _c, _r) in scanned if slot == 1), None)
                if not p1:
                    data = ItemMarketData(item=item2, required_qty=required_qty, lots=[], total_available=None,
                                          min_cost=None, reason="cannot read slot1 price in pure simulate")
                    items_data.append(data)
                    log.info("Item data incomplete => skip quest. item=%r reason=%s", item2.name, data.reason)
                    return

                est = int(p1) * required_qty
                data = ItemMarketData(item=item2, required_qty=required_qty, lots=[], total_available=None,
                                      min_cost=est, reason="pure simulate: estimate p1*need (no qty reads)")
                items_data.append(data)
                total_min_cost += est
                continue

            # нормальный режим (RUN или simulate_ui_actions): строим план с qty
            plan = build_plan_one_page(ui, required_qty, reward_total, page=1)
            if not plan or plan.remaining > 0:
                data = ItemMarketData(item=item2, required_qty=required_qty, lots=[], total_available=None,
                                      min_cost=None,
                                      reason="cannot build plan on first page (qty/price mismatch or insufficient)")
                items_data.append(data)
                log.info("Item plan failed => skip quest. item=%r reason=%s", item2.name, data.reason)
                return

            plans_by_item[item2.name] = plan
            data = ItemMarketData(item=item2, required_qty=required_qty, lots=[], total_available=None,
                                  min_cost=plan.total_cost, reason=None)
            items_data.append(data)
            total_min_cost += plan.total_cost

        log.info("Quest cost estimate: %d | reward_total=%d | delta=%d", total_min_cost, reward_total,
                 reward_total - total_min_cost)
        if reward_total < total_min_cost:
            log.info("Decision: SKIP (not profitable).")
            return

        log.info("Decision: PROFITABLE => proceed.")

        if self.ctx.mode == Mode.SIMULATE and not cfg.runtime.simulate_ui_actions:
            log.info("SIMULATE mode: no inputs. Would buy items: %s", [d.item.name for d in items_data])
            return

        # -------- Stage B: покупки --------
        for d in items_data:
            if d.item.use_image:
                # старый путь для image (там next_page)
                res = self.buy_item(d.item, d.required_qty)
                log.info(
                    "Purchase result: item=%r required=%d bought=%d spent=%d success=%s reason=%s",
                    res.item.name, res.required_qty, res.bought_qty, res.spent, res.success, res.reason
                )
                plan_for_item = plans_by_item.get(res.item.name)
                planned_cost = None
                if plan_for_item is not None:
                    planned_cost = plan_for_item.total_cost
                else:
                    # fallback to ItemMarketData.min_cost if available in items_data
                    found = next((d for d in items_data if d.item.name == res.item.name), None)
                    planned_cost = getattr(found, "min_cost", None)

                # compute reward share
                if total_min_cost and total_min_cost > 0 and planned_cost:
                    reward_share = int(round(reward_total * (planned_cost / total_min_cost)))
                else:
                    # fallback equal split by number of items
                    reward_share = int(round(reward_total / max(1, len(items_data))))

                self.purchase_logger.record(res.item.name, res.spent, reward_share)
                continue

            # обычный путь: выполнить план, покупая всегда из slot=1
            plan = plans_by_item.get(d.item.name)
            if plan is None:
                # на практике это случится только в pure simulate; тут покупать нечего
                log.info("No plan stored for item=%r => skip buy.", d.item.name)
                continue

            if self.ctx.mode == Mode.SIMULATE and cfg.runtime.simulate_ui_actions:
                log.info("[SIM-UI] Performing navigation but will NOT click BUY. item=%r", d.item.name)

            # перед покупкой ещё раз поиск (чтобы точно стоять на нужном списке)
            last_q = getattr(ui, "_last_search_query", None)

            if d.item.use_image:
                ui.run_search(d.item.name)
            else:
                if last_q != d.item.name:
                    ui.run_search(d.item.name)

            ok, spent, bought = execute_plan_one_page(ui, plan, do_buy=do_buy)
            res = PurchaseResult(
                item=d.item,
                required_qty=d.required_qty,
                bought_qty=bought,
                spent=spent,
                success=ok and (bought == d.required_qty),
                reason=None if (ok and bought == d.required_qty) else "plan execution failed / partial buy",
            )

            log.info(
                "Purchase result: item=%r required=%d bought=%d spent=%d success=%s reason=%s",
                res.item.name, res.required_qty, res.bought_qty, res.spent, res.success, res.reason
            )
            plan_for_item = plans_by_item.get(res.item.name)
            planned_cost = None
            if plan_for_item is not None:
                planned_cost = plan_for_item.total_cost
            else:
                # fallback to ItemMarketData.min_cost if available in items_data
                found = next((d for d in items_data if d.item.name == res.item.name), None)
                planned_cost = getattr(found, "min_cost", None)

            # compute reward share
            if total_min_cost and total_min_cost > 0 and planned_cost:
                reward_share = int(round(reward_total * (planned_cost / total_min_cost)))
            else:
                # fallback equal split by number of items
                reward_share = int(round(reward_total / max(1, len(items_data))))

            self.purchase_logger.record(res.item.name, res.spent, reward_share)

    def collect_image_item_data(self, item: QuestItem, required_qty: int, budget_left: int) -> ItemMarketData:
        log = self.ctx.logger
        ui = self.ctx.ui
        cfg = self.ctx.cfg

        log.info("Collect/Plan(image) item: %r required_qty=%d budget_left=%d", item.name, required_qty, budget_left)

        can_open_modals = (self.ctx.mode == Mode.RUN) or cfg.runtime.simulate_ui_actions

        if self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions:
            ui.run_search(item.name)

        def ocr_price(slot: int, page: int) -> Optional[int]:
            rect = ui.price_rect_for_slot(slot)
            crop = ui.screen.grab_region_bgr(rect)
            r = ui.ocr.read_price(
                crop,
                timeout_s=cfg.timing.wait_ocr_timeout,
                variants=cfg.ocr.variants,
                scales=cfg.ocr.scale_factors,
            )
            if r.value is None:
                log.warning("Price OCR failed: item=%r page=%d slot=%d raw=%r", item.name, page, slot, r.raw_text)
                return None
            return int(r.value)

        remain = required_qty
        cost = 0
        total_available = 0
        lots: list[LotInfo] = []

        page = 1
        max_pages = cfg.runtime.max_pages if can_open_modals else 1

        while page <= max_pages and remain > 0:
            matches = ui.find_item_slots_by_icon(item.name)  # [(slot, score), ...] отсортированы по slot
            if not matches:
                if page < max_pages:
                    ui.next_page()
                    page += 1
                    continue
                break

            # идём сверху вниз; ниже — только дороже
            progressed_on_page = False
            for slot, score in matches:
                price = ocr_price(slot, page)
                if price is None:
                    continue

                # РАННИЙ ОТСЕВ:
                # даже если весь остаток купить по текущей цене (а дальше только дороже),
                # то в бюджет уже не помещаемся => дальше смотреть бессмысленно
                if cost + price * remain > budget_left:
                    return ItemMarketData(
                        item=item,
                        required_qty=required_qty,
                        lots=lots,
                        total_available=total_available,
                        min_cost=None,
                        reason=f"unprofitable: cost={cost} price={price} remain={remain} budget_left={budget_left}",
                    )

                if not can_open_modals:
                    # в pure-simulate мы не можем читать qty — даём оптимистичную оценку и выходим
                    est = cost + price * remain
                    return ItemMarketData(
                        item=item,
                        required_qty=required_qty,
                        lots=lots,
                        total_available=None,
                        min_cost=est,
                        reason="no modal access; estimate by current best price",
                    )

                # читаем qty только если по цене ещё есть смысл
                ui.open_lot_modal(slot)
                buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
                if not buy_btn or not cancel_btn:
                    ui.close_modal_safely()
                    continue

                qty = ui.read_seller_qty_from_modal(buy_btn)
                ui.click_cancel(cancel_btn)

                if not qty or qty <= 0:
                    continue

                progressed_on_page = True
                total_available += qty

                take = min(remain, qty)
                cost += take * price
                remain -= take

                lots.append(LotInfo(page=page, slot=slot, price=price, seller_qty=qty, match_score=score))

                if remain <= 0:
                    return ItemMarketData(
                        item=item,
                        required_qty=required_qty,
                        lots=lots,
                        total_available=total_available,
                        min_cost=cost,
                        reason=None,
                    )

            # если на странице вообще не смогли продвинуться — перелистываем (или выходим)
            if remain > 0:
                if page < max_pages and can_open_modals:
                    ui.next_page()
                    page += 1
                    continue
                break

        # если дошли сюда — не хватило количества
        return ItemMarketData(
            item=item,
            required_qty=required_qty,
            lots=lots,
            total_available=total_available if total_available > 0 else 0,
            min_cost=None,
            reason=f"insufficient supply: need={required_qty} have={total_available}",
        )

    def _apply_image_mode_if_configured(self, item: QuestItem) -> QuestItem:
        """
        If config.templates.item_icons contains this item, we auto-enable use_image.
        """
        if item.use_image:
            return item
        if item.name in self.ctx.cfg.templates.item_icons:
            return QuestItem(name=item.name, qty_per_set=item.qty_per_set, use_image=True, image_key=item.name)
        return item

    def collect_item_data(self, item: QuestItem, required_qty: int) -> ItemMarketData:
        """
        Collect lots across up to max_pages:
          - for normal items: OCR prices per slot
          - for image items: find icon -> determine slot -> OCR price for that slot
        Quantities (seller availability) require opening modal:
          - only possible when RUN or simulate_ui_actions True
          - otherwise total_available/min_cost computed as best-effort (price1 * required_qty)
        """
        log = self.ctx.logger
        ui = self.ctx.ui
        cfg = self.ctx.cfg

        log.info("Collect item: %r required_qty=%d use_image=%s", item.name, required_qty, item.use_image)

        if self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions:
            ui.run_search(item.name)

        lots: list[LotInfo] = []

        can_open_modals = (self.ctx.mode == Mode.RUN) or cfg.runtime.simulate_ui_actions

        # If we cannot navigate at all (pure simulate), we try scanning current page only.
        pages_to_scan = cfg.runtime.max_pages if (
                    item.use_image and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions)) else 1

        for page in range(1, pages_to_scan + 1):
            if item.use_image:
                matches = ui.find_item_slots_by_icon(item.name)  # [(slot, score), ...]
                if not matches:
                    if page < pages_to_scan and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                        ui.next_page()
                        continue
                    break

                for slot, score in matches:
                    rect = ui.price_rect_for_slot(slot)
                    crop = ui.screen.grab_region_bgr(rect)
                    r = ui.ocr.read_price(
                        crop,
                        timeout_s=cfg.timing.wait_ocr_timeout,
                        variants=cfg.ocr.variants,
                        scales=cfg.ocr.scale_factors,
                    )
                    if r.value is None:
                        log.warning("Price OCR failed: item=%r page=%d slot=%d", item.name, page, slot)
                        continue
                    lots.append(LotInfo(page=page, slot=slot, price=r.value, seller_qty=None, match_score=score))

                # ВАЖНО: в стадии оценки НЕ открываем модалки и не читаем qty

            # read seller qty per lot (optional, best effort)
            if can_open_modals and lots and (not item.use_image):
                # only for lots on this page
                page_lots = [l for l in lots if l.page == page]
                for l in page_lots:
                    try:
                        ui.open_lot_modal(l.slot)
                        buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
                        if buy_btn is None or cancel_btn is None:
                            log.warning("Cannot locate modal buttons; skip qty read.")
                            continue
                        qty = ui.read_seller_qty_from_modal(buy_btn)
                        # always cancel in data stage
                        ui.click_cancel(cancel_btn)

                        # patch in place (recreate LotInfo)
                        lots[lots.index(l)] = LotInfo(
                            page=l.page, slot=l.slot, price=l.price, seller_qty=qty, match_score=l.match_score
                        )
                    except Exception as e:
                        log.warning("Qty read failed (skip lot). item=%r page=%d slot=%d err=%s", item.name, l.page, l.slot, e)

            if page < pages_to_scan and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                ui.next_page()

        # Compute minimal cost to fulfill required_qty
        if not lots:
            return ItemMarketData(
                item=item,
                required_qty=required_qty,
                lots=[],
                total_available=0,
                min_cost=None,
                reason="no lots detected (icon not found / OCR failed / wrong UI state)",
            )

        # Sort by price ascending (top is cheaper, but we don't rely on UI order)
        lots_sorted = sorted(lots, key=lambda x: x.price)

        known_qty = all(l.seller_qty is not None for l in lots_sorted)
        if not known_qty:
            # Best-effort: assume first lot price * required_qty
            est = lots_sorted[0].price * required_qty
            return ItemMarketData(
                item=item,
                required_qty=required_qty,
                lots=lots_sorted,
                total_available=None,
                min_cost=est,
                reason="seller quantities unknown; using price(first_lot)*required_qty estimate",
            )

        # Fill from cheapest lots
        remain = required_qty
        cost = 0
        available_total = 0
        for l in lots_sorted:
            assert l.seller_qty is not None
            available_total += l.seller_qty
            take = min(remain, l.seller_qty)
            cost += take * l.price
            remain -= take
            if remain <= 0:
                break

        if remain > 0:
            return ItemMarketData(
                item=item,
                required_qty=required_qty,
                lots=lots_sorted,
                total_available=available_total,
                min_cost=None,
                reason=f"insufficient supply: need={required_qty} have={available_total}",
            )

        return ItemMarketData(
            item=item,
            required_qty=required_qty,
            lots=lots_sorted,
            total_available=available_total,
            min_cost=cost,
            reason=None,
        )

    def buy_item(self, item: QuestItem, required_qty: int) -> PurchaseResult:
        log = self.ctx.logger
        ui = self.ctx.ui
        cfg = self.ctx.cfg

        spent = 0
        bought = 0
        remain = required_qty

        do_buy = (self.ctx.mode == Mode.RUN)

        # For image-mode we navigate pages, so do one search up-front.
        if self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions:
            ui.run_search(item.name)

        # ---------------- IMAGE MODE ----------------
        if item.use_image:
            page = 1
            while remain > 0 and page <= cfg.runtime.max_pages:
                matches = ui.find_item_slots_by_icon(item.name)  # [(slot, score), ...] sorted by slot asc
                if not matches:
                    log.info("Image item not found on page %d, going next.", page)
                    if page < cfg.runtime.max_pages and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                        ui.next_page()
                        page += 1
                        continue
                    break

                progressed = False

                # try from cheapest/top to more expensive on this page
                for slot_idx, _score in matches:
                    if remain <= 0:
                        break

                    # OCR price for this slot (logging/spent)
                    rect = ui.price_rect_for_slot(slot_idx)
                    crop = ui.screen.grab_region_bgr(rect)
                    pr = ui.ocr.read_price(
                        crop,
                        timeout_s=cfg.timing.wait_ocr_timeout,
                        variants=cfg.ocr.variants,
                        scales=cfg.ocr.scale_factors,
                    )
                    if pr.value is None:
                        log.warning("Cannot OCR price before buy. page=%d slot=%d => skip slot", page, slot_idx)
                        continue
                    price = pr.value

                    ui.open_lot_modal(slot_idx)
                    buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
                    if buy_btn is None or cancel_btn is None:
                        log.warning("Modal buttons not found => skip slot.")
                        ui.close_modal_safely()
                        continue

                    seller_qty = ui.read_seller_qty_from_modal(buy_btn)
                    if not seller_qty or seller_qty <= 0:
                        ui.click_cancel(cancel_btn)
                        continue

                    to_buy = min(remain, seller_qty)
                    log.info(
                        "Buy attempt: item=%r page=%d slot=%d price=%d seller_qty=%d to_buy=%d remain=%d",
                        item.name, page, slot_idx, price, seller_qty, to_buy, remain
                    )

                    # SIM-UI mode: never click BUY
                    if self.ctx.mode == Mode.SIMULATE and cfg.runtime.simulate_ui_actions:
                        ui.click_cancel(cancel_btn)
                        bought += to_buy
                        spent += to_buy * price
                        remain -= to_buy
                        progressed = True
                        break

                    ok = ui.set_buy_quantity(buy_btn, to_buy)
                    if not ok:
                        log.warning("Failed to set buy quantity => cancel and try next slot.")
                        ui.click_cancel(cancel_btn)
                        continue

                    if do_buy:
                        ui.click_buy(buy_btn)
                    else:
                        ui.click_cancel(cancel_btn)

                    bought += to_buy
                    spent += to_buy * price
                    remain -= to_buy
                    progressed = True

                    # list may shift after buy => re-scan from the top (cheapest first)
                    ui.inp.sleep(cfg.timing.wait_open_modal * 0.4)
                    break

                if remain <= 0:
                    break

                if progressed:
                    # re-scan same page to keep taking cheapest first
                    continue

                # couldn't buy anything on this page => try next page
                if page < cfg.runtime.max_pages and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                    ui.next_page()
                    page += 1
                    continue

                break

            success = (remain <= 0)
            reason = None if success else f"not enough supply or OCR/template failures; remain={remain}"
            return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=success, reason=reason)

        # ---------------- TEXT/OCR MODE (fallback) ----------------
        page = 1
        while remain > 0 and page <= cfg.runtime.max_pages:
            if self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions:
                ui.run_search(item.name)

            slot_idx = 1

            rect = ui.price_rect_for_slot(slot_idx)
            crop = ui.screen.grab_region_bgr(rect)
            pr = ui.ocr.read_price(
                crop, timeout_s=cfg.timing.wait_ocr_timeout, variants=cfg.ocr.variants, scales=cfg.ocr.scale_factors
            )
            if pr.value is None:
                log.warning("Cannot OCR price before buy. page=%d slot=%d => skip/next", page, slot_idx)
                if page < cfg.runtime.max_pages and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                    ui.next_page()
                    page += 1
                    continue
                break
            price = pr.value

            ui.open_lot_modal(slot_idx)
            buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
            if buy_btn is None or cancel_btn is None:
                log.warning("Modal buttons not found => cancel/skip.")
                ui.close_modal_safely()
                return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=False, reason="modal buttons not found")

            seller_qty = ui.read_seller_qty_from_modal(buy_btn)
            if seller_qty is None or seller_qty <= 0:
                log.info("Seller qty unknown/0 => cancel and next lot/page.")
                ui.click_cancel(cancel_btn)
                if page < cfg.runtime.max_pages and (self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                    ui.next_page()
                    page += 1
                    continue
                break

            to_buy = min(remain, seller_qty)
            log.info(
                "Buy attempt: item=%r page=%d slot=%d price=%d seller_qty=%d to_buy=%d remain=%d",
                item.name, page, slot_idx, price, seller_qty, to_buy, remain
            )

            if self.ctx.mode == Mode.SIMULATE and cfg.runtime.simulate_ui_actions:
                ui.click_cancel(cancel_btn)
                bought += to_buy
                spent += to_buy * price
                remain -= to_buy
                continue

            ok = ui.set_buy_quantity(buy_btn, to_buy)
            if not ok:
                ui.click_cancel(cancel_btn)
                return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=False, reason="cannot set buy qty")

            ui.click_buy(buy_btn)
            bought += to_buy
            spent += to_buy * price
            remain -= to_buy

            ui.inp.sleep(cfg.timing.wait_open_modal * 0.6)

        success = (remain <= 0)
        reason = None if success else f"not enough supply or OCR/template failures; remain={remain}"
        return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=success, reason=reason)
