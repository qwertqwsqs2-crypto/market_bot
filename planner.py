from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models import Point
from ui import MarketUI


@dataclass(frozen=True)
class PlanStep:
    slot: int          # слот, из которого планировали взять (для логов/отладки)
    price: int
    take: int          # сколько купить на этом шаге


@dataclass(frozen=True)
class PlanResult:
    steps: list[PlanStep]
    total_cost: int
    remaining: int     # 0 если план успешен


def build_plan_one_page(
    ui: MarketUI,
    need_total: int,
    reward_total: int,
    *,
    page: int = 1,
) -> Optional[PlanResult]:
    """
    Фаза 1: строим план закупки на ОДНОЙ странице.
    Слоты 1..8: для каждого слота сначала проверяем min_possible,
    и только потом (если выгодно) открываем модалку и читаем qty.
    """
    prices_raw = ui.scan_prices_on_page(page=page)
    price_by_slot: dict[int, int] = {slot: price for (slot, price, _conf, _raw) in prices_raw}

    remaining = need_total
    cost_so_far = 0
    steps: list[PlanStep] = []

    # если цена первого слота не читается — смысла продолжать нет
    p1 = price_by_slot.get(1, 0)
    if p1 <= 0:
        ui.logger.info("Plan: slot1 price is not readable -> skip")
        return None

    for slot in range(1, ui.cfg.runtime.max_slots + 1):
        if remaining <= 0:
            break

        p = price_by_slot.get(slot, 0)
        if p <= 0:
            continue

        # ключевая проверка (как ты описал):
        # "если даже остаток * текущая цена не влезает, дальше только дороже => план невозможен"
        min_possible = cost_so_far + remaining * p
        if min_possible > reward_total:
            ui.logger.info(
                "Plan: not affordable at slot=%d price=%d -> min_possible=%d > reward=%d",
                slot, p, min_possible, reward_total
            )
            return None

        # читаем qty у продавца (только если по цене еще имеет смысл)
        ui.open_lot_modal(slot_index_1based=slot)
        buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
        if not buy_btn:
            ui.logger.warning("Plan: buy button not found in modal (slot=%d).", slot)
            ui.close_modal_safely()
            continue

        avail = ui.read_seller_qty_from_modal(buy_btn)
        # для чтения qty модалку можно закрыть обычным cancel (модалка точно открыта)
        if cancel_btn:
            ui.click_cancel(cancel_btn)
        else:
            ui.close_modal_safely()

        if not avail or avail <= 0:
            continue

        take = min(avail, remaining)
        steps.append(PlanStep(slot=slot, price=p, take=take))

        cost_so_far += take * p
        remaining -= take

    return PlanResult(steps=steps, total_cost=cost_so_far, remaining=remaining)


def execute_plan_one_page(
    ui: MarketUI,
    plan: PlanResult,
    *,
    do_buy: bool,
) -> tuple[bool, int, int]:
    """
    Фаза 2: выполняем план.

    RUN:
      - покупаем всегда из slot=1 (список сдвигается вверх, т.к. мы выкупаем весь доступный qty каждого слота)
      - устанавливаем qty и жмём BUY

    SIMULATE (do_buy=False):
      - список НЕ сдвигается, поэтому открываем ИМЕННО плановый slot (step.slot)
      - НЕ вводим qty, НЕ жмём BUY (только лог + закрыть модалку)
    """
    spent = 0
    bought = 0

    for step_idx, step in enumerate(plan.steps, start=1):
        ui.logger.info(
            "BUY step %d/%d: planned slot=%d take=%d price=%d",
            step_idx, len(plan.steps), step.slot, step.take, step.price
        )

        # slot selection differs for RUN vs SIM
        open_slot = 1 if do_buy else step.slot
        ui.open_lot_modal(slot_index_1based=open_slot)

        buy_btn, cancel_btn = ui.locate_buy_cancel_buttons()
        if not buy_btn:
            ui.logger.warning("Buy: buy button not found in modal on step=%d.", step_idx)
            ui.close_modal_safely()
            return False, spent, bought

        avail = ui.read_seller_qty_from_modal(buy_btn)
        if not avail or avail < step.take:
            ui.logger.warning(
                "Buy: qty mismatch on step=%d. avail=%s needed=%d. Abort.",
                step_idx, avail, step.take
            )
            ui.close_modal_safely()
            return False, spent, bought

        if do_buy:
            ok = ui.set_buy_quantity(buy_btn, step.take)  # <-- теперь bool
            if not ok:
                ui.logger.warning("Buy: cannot set quantity reliably. Abort step=%d.", step_idx)
                ui.close_modal_safely()
                return False, spent, bought

            ui.click_buy(buy_btn)
            ui.inp.sleep(ui.cfg.timing.wait_after_buy)
        else:
            ui.logger.info("[SIM] Would set qty=%d and click BUY now.", step.take)

        spent += step.take * step.price
        bought += step.take

        # close modal
        if cancel_btn:
            ui.click_cancel(cancel_btn)
            ui.inp.sleep(ui.cfg.timing.wait_after_cancel)
        else:
            ui.close_modal_safely()

        ui.inp.sleep(ui.cfg.timing.wait_between_plan_steps)

    return True, spent, bought

