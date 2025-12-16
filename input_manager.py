from __future__ import annotations

import time
import logging
import pyautogui  # type: ignore

logger = logging.getLogger(__name__)

try:
    import pyperclip  # type: ignore
    HAS_PYPERCLIP = True
except Exception:
    HAS_PYPERCLIP = False

try:
    import win32api  # type: ignore
    import win32con  # type: ignore
    import win32clipboard  # type: ignore
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False


class InputManager:
    # ---------- Mouse ----------
    @staticmethod
    def human_click(x: int, y: int, duration: float = 0.1):
        try:
            pyautogui.moveTo(x, y, duration=duration)
            time.sleep(0.04)
            pyautogui.mouseDown()
            time.sleep(0.06)
            pyautogui.mouseUp()
            time.sleep(0.08)
        except Exception as e:
            logger.error(f"human_click error: {e}")

    @staticmethod
    def human_double_click(x: int, y: int):
        try:
            InputManager.human_click(x, y)
            time.sleep(0.05)
            pyautogui.mouseDown()
            time.sleep(0.05)
            pyautogui.mouseUp()
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"human_double_click error: {e}")

    # ---------- Keyboard ----------
    @staticmethod
    def key_combo(vk1: int, vk2: int):
        """
        Например: Ctrl+A / Ctrl+C / Ctrl+V
        """
        try:
            if HAS_WIN32:
                win32api.keybd_event(vk1, 0, 0, 0)
                time.sleep(0.03)
                win32api.keybd_event(vk2, 0, 0, 0)
                time.sleep(0.03)
                win32api.keybd_event(vk2, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.03)
                win32api.keybd_event(vk1, 0, win32con.KEYEVENTF_KEYUP, 0)
            else:
                pyautogui.hotkey('ctrl', chr(vk2).lower())
        except Exception as e:
            logger.error(f"key_combo error: {e}")

    # ---------- Clipboard ----------
    @staticmethod
    def set_clipboard(text: str) -> bool:
        try:
            text = str(text)

            if HAS_PYPERCLIP:
                pyperclip.copy(text)
                return True

            if HAS_WIN32:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
                win32clipboard.CloseClipboard()
                return True

            logger.error("Clipboard set failed: no backend")
            return False

        except Exception as e:
            logger.error(f"set_clipboard error: {e}")
            return False

    @staticmethod
    def get_clipboard() -> str:
        try:
            if HAS_PYPERCLIP:
                return pyperclip.paste()

            if HAS_WIN32:
                win32clipboard.OpenClipboard()
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                win32clipboard.CloseClipboard()
                return data

            logger.error("Clipboard get failed: no backend")
            return ""

        except Exception as e:
            logger.error(f"get_clipboard error: {e}")
            return ""
