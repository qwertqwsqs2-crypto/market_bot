# main.py (обновлён)
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
import sys

from bot import BotContext, MarketBot, parse_quests_db  # type: ignore
from config import AppConfig  # type: ignore
from input_controller import GameInput, SimulatedInput  # type: ignore
from logger import setup_logging  # type: ignore
from ocr_utils import OCRReader  # type: ignore
from ui import MarketUI  # type: ignore
from quests_db import ALL_QUESTS  # type: ignore
from models import Mode  # type: ignore

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Market quest items auto-buyer (OCR + template matching).")
    p.add_argument("--level", type=int, choices=[60, 65], help="Quest level: 60 or 65")
    p.add_argument("--sets", type=int, help="Number of sets (integer)")
    p.add_argument("--mode", type=str, choices=[m.value for m in Mode], help="simulate or run")
    p.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (INFO, DEBUG, ...)")
    p.add_argument("--i-understand", action="store_true", help="Required for run mode: acknowledge responsibility/ToS risk.")
    p.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.gui:
        # Launch GUI
        try:
            from gui.main_window import MainWindow  # type: ignore
        except Exception as e:
            print("GUI import failed:", e)
            sys.exit(1)
        app = MainWindow(config_path=args.config)
        app.mainloop()
        return

    # CLI mode (existing flow)
    if args.level is None or args.sets is None or args.mode is None:
        print("For CLI mode you must specify --level, --sets and --mode (or use --gui).")
        return

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logger = setup_logging(level=log_level)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s (copy config.example.json -> config.json)", cfg_path)
        return

    cfg = AppConfig.load(cfg_path)

    mode = Mode(str(args.mode))
    level = int(args.level)
    sets = int(args.sets)

    if sets <= 0:
        logger.error("sets must be positive.")
        return

    # Build input controller
    if mode == Mode.SIMULATE and not cfg.runtime.simulate_ui_actions:
        inp = SimulatedInput(logger=logger)
    else:
        if cfg.runtime.input_backend != "pyautogui":
            logger.warning("Unknown input_backend=%r; fallback to pyautogui.", cfg.runtime.input_backend)
        inp = GameInput(wait_between_clicks=cfg.timing.wait_between_clicks, logger=logger)

    ocr = OCRReader(
        use_easyocr=cfg.ocr.use_easyocr,
        langs=cfg.ocr.easyocr_langs,
        use_tesseract=cfg.ocr.use_tesseract_fallback,
        min_conf=cfg.ocr.min_confidence,
        prefer_gpu=cfg.ocr.easyocr_gpu,
    )

    ui = MarketUI(cfg=cfg, inp=inp, ocr=ocr, logger=logger)

    quests = parse_quests_db(ALL_QUESTS)
    # Prepare events/stop/pause for CLI too (optional)
    import threading, queue
    stop_event = threading.Event()
    pause_event = threading.Event()
    events_q = queue.Queue()

    ctx = BotContext(cfg=cfg, ui=ui, logger=logger, mode=mode, level=level, sets=sets,
                     stop_event=stop_event, pause_event=pause_event, events=events_q)
    bot = MarketBot(ctx)
    startup_delay = cfg.timing.startup_delay
    logger.info("Startup delay %.1fs — переключитесь на окно игры.", startup_delay)
    time.sleep(startup_delay)
    try:
        bot.run(quests)
    finally:
        ui.cleanup()


if __name__ == "__main__":
    main()
