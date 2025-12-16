from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(str, Enum):
    SIMULATE = "simulate"
    RUN = "run"


class Category(str, Enum):
    TROPHY = "Трофеи"
    RESOURCES = "Ресурсы"


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def moved(self, dx: int, dy: int) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.w, self.h)


@dataclass(frozen=True)
class QuestItem:
    name: str
    qty_per_set: int
    # For ambiguous names: identify slot by icon template (template matching)
    use_image: bool = False
    # key in config.templates.item_icons or direct path
    image_key: Optional[str] = None


@dataclass(frozen=True)
class QuestDefinition:
    quest: str
    category: Category
    reward60: int
    reward65: int
    items: list[QuestItem]


@dataclass(frozen=True)
class LotInfo:
    page: int
    slot: int  # 1..8
    price: int
    seller_qty: Optional[int] = None
    match_score: Optional[float] = None


@dataclass(frozen=True)
class ItemMarketData:
    item: QuestItem
    required_qty: int
    lots: list[LotInfo]
    total_available: Optional[int]  # None if unknown
    min_cost: Optional[int]         # None if cannot be computed
    reason: Optional[str] = None


@dataclass(frozen=True)
class PurchaseResult:
    item: QuestItem
    required_qty: int
    bought_qty: int
    spent: int
    success: bool
    reason: Optional[str] = None
