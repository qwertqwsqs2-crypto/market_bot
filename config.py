from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from models import Point, Rect


@dataclass(frozen=True)
class TimingConfig:
    startup_delay: float = 3.0
    wait_between_clicks: float = 0.12
    wait_open_modal: float = 0.9
    # How long to wait after typing into the search input for the list to refresh.
    wait_after_search: float = 0.7
    wait_ocr_timeout: float = 2.0
    wait_page_flip: float = 1.0



@dataclass(frozen=True)
class OffsetsConfig:
    # Required offsets (relative to anchor)
    anchor_search: Point = Point(-312, 65)
    cat_trophy: Point = Point(-218, 379)
    cat_res: Point = Point(-228, 484)

    first_slot: Point = Point(62, 145)
    region_price: Rect = Rect(166, 118, 140, 44)

    buy_input_offset: Point = Point(180, -87)
    buy_btn_fallback: Point = Point(-11, 293)
    buy_cancel_fallback: Point = Point(170, 211)
    next_page: Point = Point(30, 605)

    slot_height: int = 59

    # Optional: region covering the whole list for template scanning of item icons.
    # If None, we will derive a best-effort region from first_slot/slot_height.
    region_slots: Optional[Rect] = None


@dataclass(frozen=True)
class OCRConfig:
    use_easyocr: bool = True
    easyocr_langs: list[str] = field(default_factory=lambda: ["ru", "en"])
    easyocr_gpu: bool = True
    use_tesseract_fallback: bool = True

    # For price OCR robustness
    min_confidence: float = 0.35
    # how many preprocessing variants to try
    variants: int = 5
    scale_factors: list[int] = field(default_factory=lambda: [2, 3])


@dataclass(frozen=True)
class TemplateConfig:
    # Base folder with templates (anchor, buttons, item icons)
    templates_dir: Path = Path("templates")

    anchor_market: str = "anchor_market.png"
    btn_buy: str = "btn_buy.png"
    btn_cancel: str = "btn_cancel.png"
    btn_next: str = "btn_next.png"

    # Item icons for "use_image" items:
    # {"Item Name": "items/my_icon.png"}
    item_icons: dict[str, str] = field(default_factory=dict)

    # Matching thresholds
    threshold_anchor: float = 0.80
    threshold_buttons: float = 0.82
    threshold_item_icon: float = 0.86
    multi_scale: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    max_pages: int = 10
    max_slots: int = 8

    # Simulate behavior:
    # - simulate mode by default does not send inputs; it only logs intended actions.
    # If you set this True, simulate will navigate UI but never clicks "Buy"
    # (it will still open modals and press Cancel).
    simulate_ui_actions: bool = False

    # Input backend: "pyautogui" recommended.
    input_backend: str = "pyautogui"


@dataclass(frozen=True)
class SafetyConfig:
    # Must be true to allow run mode (also requires --i-understand).
    acknowledge_rights: bool = False


@dataclass(frozen=True)
class AppConfig:
    timing: TimingConfig = TimingConfig()
    offs: OffsetsConfig = OffsetsConfig()
    ocr: OCRConfig = OCRConfig()
    templates: TemplateConfig = TemplateConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    safety: SafetyConfig = SafetyConfig()

    @staticmethod
    def load(path: Path) -> "AppConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AppConfig.from_dict(raw, base_dir=path.parent)

    @staticmethod
    def from_dict(d: dict[str, Any], base_dir: Path | None = None) -> "AppConfig":
        base_dir = base_dir or Path(".")

        timing = TimingConfig(
            startup_delay=float(d.get("startup_delay", 3.0)),
            wait_between_clicks=float(d.get("wait_between_clicks", 0.12)),
            wait_open_modal=float(d.get("wait_open_modal", 0.9)),
            wait_after_search=float(d.get("wait_after_search", 0.7)),
            wait_ocr_timeout=float(d.get("wait_ocr_timeout", 5.0)),
        )

        offs_raw = d.get("offs", {})
        offs = OffsetsConfig(
            anchor_search=Point(*offs_raw.get("anchor_search", [-312, 65])),
            cat_trophy=Point(*offs_raw.get("cat_trophy", [-218, 379])),
            cat_res=Point(*offs_raw.get("cat_res", [-228, 484])),
            first_slot=Point(*offs_raw.get("first_slot", [62, 145])),
            region_price=Rect(*offs_raw.get("region_price", [166, 118, 140, 44])),
            buy_input_offset=Point(*offs_raw.get("buy_input_offset", [180, -87])),
            buy_btn_fallback=Point(*offs_raw.get("buy_btn_fallback", [-11, 293])),
            buy_cancel_fallback=Point(*offs_raw.get("buy_cancel_fallback", [170, 211])),
            next_page=Point(*offs_raw.get("next_page", [30, 605])),
            slot_height=int(offs_raw.get("slot_height", 59)),
            region_slots=Rect(*offs_raw["region_slots"]) if "region_slots" in offs_raw else None,
        )

        ocr_raw = d.get("ocr", {})
        ocr = OCRConfig(
            use_easyocr=bool(ocr_raw.get("use_easyocr", True)),
            easyocr_langs=list(ocr_raw.get("easyocr_langs", ["ru", "en"])),
            easyocr_gpu=bool(ocr_raw.get("easyocr_gpu", True)),
            use_tesseract_fallback=bool(ocr_raw.get("use_tesseract_fallback", True)),
            min_confidence=float(ocr_raw.get("min_confidence", 0.35)),
            variants=int(ocr_raw.get("variants", 3)),
            scale_factors=list(ocr_raw.get("scale_factors", [2, 3])),
        )

        tpl_raw = d.get("templates", {})
        templates_dir = Path(tpl_raw.get("templates_dir", "templates"))
        if not templates_dir.is_absolute():
            templates_dir = (base_dir / templates_dir).resolve()

        templates = TemplateConfig(
            templates_dir=templates_dir,
            anchor_market=str(tpl_raw.get("anchor_market", "anchor_market.png")),
            btn_buy=str(tpl_raw.get("btn_buy", "btn_buy.png")),
            btn_cancel=str(tpl_raw.get("btn_cancel", "btn_cancel.png")),
            btn_next=str(tpl_raw.get("btn_next", "btn_next.png")),
            item_icons=dict(tpl_raw.get("item_icons", {})),
            threshold_anchor=float(tpl_raw.get("threshold_anchor", 0.80)),
            threshold_buttons=float(tpl_raw.get("threshold_buttons", 0.82)),
            threshold_item_icon=float(tpl_raw.get("threshold_item_icon", 0.86)),
            multi_scale=bool(tpl_raw.get("multi_scale", True)),
        )

        runtime_raw = d.get("runtime", {})
        runtime = RuntimeConfig(
            max_pages=int(d.get("max_pages", runtime_raw.get("max_pages", 10))),
            max_slots=int(d.get("max_slots", runtime_raw.get("max_slots", 8))),
            simulate_ui_actions=bool(runtime_raw.get("simulate_ui_actions", False)),
            input_backend=str(runtime_raw.get("input_backend", "pyautogui")),
        )

        safety_raw = d.get("safety", {})
        safety = SafetyConfig(
            acknowledge_rights=bool(safety_raw.get("acknowledge_rights", False))
        )

        return AppConfig(
            timing=timing,
            offs=offs,
            ocr=ocr,
            templates=templates,
            runtime=runtime,
            safety=safety,
        )
