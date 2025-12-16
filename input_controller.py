from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import pyautogui  # type: ignore
try:
    import win32con  # type: ignore
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

from models import Point
from input_manager import InputManager as I


class InputController(Protocol):
    def click(self, p: Point) -> None: ...
    def dblclick(self, p: Point) -> None: ...
    def type_text(self, text: str) -> None: ...
    def press(self, key: str, presses: int = 1, interval: float = 0.01) -> None: ...
    def paste_text(self, text: str) -> None: ...
    def copy_hotkey(self) -> None: ...
    def sleep(self, seconds: float) -> None: ...
    def get_clipboard(self) -> str: ...
    def set_clipboard(self, text: str) -> bool: ...
    def press_hotkey(self, *keys: str) -> None: ...
    def ctrl_combo(self, letter: str) -> None: ...

@dataclass
class SimulatedInput(InputController):
    logger: object

    def click(self, p: Point) -> None:
        self.logger.info("[SIM] click at %s", p)

    def dblclick(self, p: Point) -> None:
        self.logger.info("[SIM] dblclick at %s", p)

    def type_text(self, text: str) -> None:
        self.logger.info("[SIM] type_text: %r", text)

    def press(self, key: str, presses: int = 1, interval: float = 0.01) -> None:
        self.logger.info("[SIM] press %s x%d", key, presses)

    def paste_text(self, text: str) -> None:
        self.logger.info("[SIM] paste_text: %r", text)

    def copy_hotkey(self) -> None:
        self.logger.info("[SIM] ctrl+c")

    def sleep(self, seconds: float) -> None:
        self.logger.info("[SIM] sleep %.2fs", seconds)

    def get_clipboard(self) -> str:
        self.logger.info("[SIM] get_clipboard")
        return ""

    def set_clipboard(self, text: str) -> bool:
        self.logger.info("[SIM] set_clipboard: %r", text)
        return True

    def press_hotkey(self, *keys: str) -> None:
        self.logger.info("[SIM] hotkey: %s", "+".join(keys))

    def ctrl_combo(self, letter: str) -> None:
        self.logger.info("[SIM] ctrl_combo: %r", letter)


class GameInput(InputController):
    def __init__(self, logger: object, wait_between_clicks: float) -> None:
        self.logger = logger
        self.wait_between_clicks = float(wait_between_clicks)
        pyautogui.FAILSAFE = True

    def click(self, p: Point) -> None:
        I.human_click(p.x, p.y)

    def dblclick(self, p: Point) -> None:
        I.human_double_click(p.x, p.y)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def type_text(self, text: str) -> None:
        pyautogui.typewrite(text, interval=0.01)

    def press(self, key: str, presses: int = 1, interval: float = 0.01) -> None:
        pyautogui.press(key, presses=presses, interval=interval)

    def set_clipboard(self, text: str) -> bool:
        return I.set_clipboard(text)

    def get_clipboard(self) -> str:
        return I.get_clipboard()

    def ctrl_combo(self, letter: str) -> None:
        vk_ctrl = win32con.VK_CONTROL if HAS_WIN32 else 0x11
        I.key_combo(vk_ctrl, ord(letter.upper()))

    def press_hotkey(self, *keys: str) -> None:
        ks = [k.lower() for k in keys if k]
        if ks == ["ctrl", "a"]:
            self.ctrl_combo("A"); return
        if ks == ["ctrl", "c"]:
            self.ctrl_combo("C"); return
        if ks == ["ctrl", "v"]:
            self.ctrl_combo("V"); return
        pyautogui.hotkey(*ks)

    def paste_text(self, text: str) -> None:
        # "вставка" через clipboard+Ctrl+V, но без гарантии что поле поддерживает
        self.set_clipboard(text)
        self.ctrl_combo("V")

    def copy_hotkey(self) -> None:
        self.ctrl_combo("C")
