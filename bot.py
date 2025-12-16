from __future__ import annotations

from dataclasses import dataclass
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
                data = self.collect_item_data(item2, required_qty)
                items_data.append(data)

                if data.min_cost is None:
                    log.info("Item data incomplete => skip quest. item=%r reason=%s", item2.name, data.reason)
                    return

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
                found = ui.find_item_slot_by_icon(item.name)
                if not found:
                    if item.use_image and page < pages_to_scan and (
                            self.ctx.mode == Mode.RUN or cfg.runtime.simulate_ui_actions):
                        ui.next_page()
                        continue
                    break
                slot, score = found
                # OCR price for that slot
                rect = ui.price_rect_for_slot(slot)
                crop = ui.screen.grab_region_bgr(rect)
                r = ui.ocr.read_price(
                    crop,
                    timeout_s=cfg.timing.wait_ocr_timeout,
                    variants=cfg.ocr.variants,
                    scales=cfg.ocr.scale_factors,
                )
                if r.value is None:
                    log.warning("Price OCR failed for image-item slot. item=%r page=%d slot=%d", item.name, page, slot)
                else:
                    lots.append(LotInfo(page=page, slot=slot, price=r.value, seller_qty=None, match_score=score))
            else:
                scanned = ui.scan_prices_on_page(page)
                for (slot, price, _conf, _raw) in scanned:
                    lots.append(LotInfo(page=page, slot=slot, price=price))

            # read seller qty per lot (optional, best effort)
            if can_open_modals and lots:
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
        """
        Buying loop:
          - always re-run search / re-read page before action
          - default buys from slot 1; for image items find slot by icon each iteration
          - partial buys if seller has less than remaining
          - up to max_pages pages
        In simulate_ui_actions: it navigates & opens modals but never presses BUY.
        """
        log = self.ctx.logger
        ui = self.ctx.ui
        cfg = self.ctx.cfg

        spent = 0
        bought = 0
        remain = required_qty

        for page in range(1, cfg.runtime.max_pages + 1):
            if remain <= 0:
                break

            ui.run_search(item.name)

            # decide which slot to open
            if item.use_image:
                found = ui.find_item_slot_by_icon(item.name)
                if not found:
                    log.info("Image item not found on page %d, going next.", page)
                    if page < cfg.runtime.max_pages:
                        ui.next_page()
                        continue
                    break
                slot_idx, _score = found
            else:
                slot_idx = 1

            # Read current price for this slot (for logging/spent)
            rect = ui.price_rect_for_slot(slot_idx)
            crop = ui.screen.grab_region_bgr(rect)
            pr = ui.ocr.read_price(
                crop, timeout_s=cfg.timing.wait_ocr_timeout, variants=cfg.ocr.variants, scales=cfg.ocr.scale_factors
            )
            if pr.value is None:
                log.warning("Cannot OCR price before buy. page=%d slot=%d => skip/next", page, slot_idx)
                if page < cfg.runtime.max_pages:
                    ui.next_page()
                    continue
                break
            price = pr.value

            ui.open_lot_modal(slot_idx)
            buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
            if buy_btn is None or cancel_btn is None:
                log.warning("Modal buttons not found => cancel/skip.")
                # best-effort click cancel fallback
                if cancel_btn:
                    ui.click_cancel(cancel_btn)
                return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=False, reason="modal buttons not found")

            seller_qty = ui.read_seller_qty_from_modal(buy_btn)
            if seller_qty is None or seller_qty <= 0:
                log.info("Seller qty unknown/0 => cancel and next lot/page.")
                ui.click_cancel(cancel_btn)
                if page < cfg.runtime.max_pages:
                    ui.next_page()
                    continue
                break

            to_buy = min(remain, seller_qty)
            log.info("Buy attempt: item=%r page=%d slot=%d price=%d seller_qty=%d to_buy=%d remain=%d",
                     item.name, page, slot_idx, price, seller_qty, to_buy, remain)

            # SIM-UI mode: never click BUY
            if self.ctx.mode == Mode.SIMULATE and cfg.runtime.simulate_ui_actions:
                ui.click_cancel(cancel_btn)
                # pretend we could buy (just for trace)
                bought += to_buy
                spent += to_buy * price
                remain -= to_buy
                # but do not actually affect UI; continue loop
                continue

            ui.set_buy_quantity(buy_btn, to_buy)
            ui.click_buy(buy_btn)
            bought += to_buy
            spent += to_buy * price
            remain -= to_buy

            # after buy, list shifts => next iteration re-runs search and re-finds slot
            ui.inp.sleep(cfg.timing.wait_open_modal * 0.6)

        success = (remain <= 0)
        reason = None if success else f"not enough supply or OCR/template failures; remain={remain}"
        return PurchaseResult(item=item, required_qty=required_qty, bought_qty=bought, spent=spent, success=success, reason=reason)
