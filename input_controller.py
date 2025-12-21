from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass
from typing import Protocol, Optional

import pyautogui  # type: ignore

try:
    import win32con  # type: ignore

    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

from models import Point
from input_manager import InputManager as I
from config import HumanizationConfig


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
    def __init__(self, logger: object, wait_between_clicks: float,
                 humanization_cfg: Optional[HumanizationConfig] = None) -> None:
        self.logger = logger
        self.wait_between_clicks = float(wait_between_clicks)
        self.humanization = humanization_cfg
        pyautogui.FAILSAFE = True

        # Track last mouse position for natural movements
        self._last_mouse_pos: Optional[tuple[int, int]] = None

    def _should_humanize(self) -> bool:
        """Check if humanization is enabled"""
        return self.humanization is not None and self.humanization.enabled

    def _random_delay(self) -> float:
        """Get random delay within configured range"""
        if not self._should_humanize():
            return 0.0
        return random.uniform(
            self.humanization.random_delay_min,
            self.humanization.random_delay_max
        )

    def _thinking_pause(self) -> None:
        """Occasional longer pause to simulate human thinking"""
        if not self._should_humanize():
            return

        if random.random() < self.humanization.thinking_pause_chance:
            pause = random.uniform(
                self.humanization.thinking_pause_min,
                self.humanization.thinking_pause_max
            )
            self.logger.debug("Thinking pause: %.2fs", pause)
            time.sleep(pause)

    def _humanized_duration(self) -> float:
        """Get randomized mouse movement duration"""
        if not self._should_humanize():
            return 0.1

        return random.uniform(
            self.humanization.mouse_duration_min,
            self.humanization.mouse_duration_max
        )

    def _add_mouse_offset(self, x: int, y: int) -> tuple[int, int]:
        """Add small random offset to coordinates"""
        if not self._should_humanize():
            return x, y

        offset_range = self.humanization.mouse_offset_range
        dx = random.randint(-offset_range, offset_range)
        dy = random.randint(-offset_range, offset_range)
        return x + dx, y + dy

    def _bezier_curve_points(self, start: tuple[int, int], end: tuple[int, int],
                             num_points: int = 20) -> list[tuple[int, int]]:
        """
        Generate points along a Bezier curve for natural mouse movement.
        Uses quadratic Bezier with one control point for slight curve.
        """
        x0, y0 = start
        x2, y2 = end

        # Control point: offset perpendicular to the direct path
        mid_x, mid_y = (x0 + x2) / 2, (y0 + y2) / 2
        dx, dy = x2 - x0, y2 - y0
        distance = math.sqrt(dx * dx + dy * dy)

        # Add curve proportional to distance (but not too much)
        curve_amount = min(distance * 0.15, 50)

        # Perpendicular offset
        if distance > 0:
            perp_x = -dy / distance * curve_amount * random.choice([-1, 1])
            perp_y = dx / distance * curve_amount * random.choice([-1, 1])
        else:
            perp_x, perp_y = 0, 0

        x1 = mid_x + perp_x
        y1 = mid_y + perp_y

        points = []
        for i in range(num_points + 1):
            t = i / num_points
            # Quadratic Bezier formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2
            y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2
            points.append((int(x), int(y)))

        return points

    def _humanized_move(self, target_x: int, target_y: int) -> None:
        """
        Move mouse to target with human-like curved path and variable speed.
        """
        if not self._should_humanize():
            # Simple linear movement
            duration = self._humanized_duration()
            pyautogui.moveTo(target_x, target_y, duration=duration)
            return

        # Get current position
        current_pos = pyautogui.position()
        start_x, start_y = current_pos

        # Calculate distance
        distance = math.sqrt((target_x - start_x) ** 2 + (target_y - start_y) ** 2)

        # For very short distances, just move directly
        if distance < 20:
            pyautogui.moveTo(target_x, target_y, duration=0.05)
            return

        # Generate curved path
        num_points = max(10, int(distance / 20))  # More points for longer distances
        path = self._bezier_curve_points((start_x, start_y), (target_x, target_y), num_points)

        # Move along the path with variable speed
        total_duration = self._humanized_duration()
        time_per_point = total_duration / len(path)

        for i, (px, py) in enumerate(path[1:], 1):  # Skip first point (current position)
            # Variable speed: slower at start/end, faster in middle
            progress = i / len(path)
            speed_factor = 1.0 - abs(2 * progress - 1) * 0.5  # Parabolic speed profile
            point_duration = time_per_point * speed_factor

            # Add micro-jitter for realism
            jitter_x = random.randint(-1, 1) if random.random() < 0.3 else 0
            jitter_y = random.randint(-1, 1) if random.random() < 0.3 else 0

            pyautogui.moveTo(px + jitter_x, py + jitter_y, duration=point_duration)

        # Ensure we end exactly at target
        pyautogui.moveTo(target_x, target_y, duration=0.02)

    def click(self, p: Point) -> None:
        """Humanized click with curved mouse movement and random delays"""
        # Optional thinking pause before action
        self._thinking_pause()

        # Add small random offset to click position
        click_x, click_y = self._add_mouse_offset(p.x, p.y)

        # Move with human-like curve
        self._humanized_move(click_x, click_y)

        # Small pause before click
        time.sleep(self._random_delay() + 0.04)

        # Actual click with slight randomization
        pyautogui.mouseDown()
        time.sleep(random.uniform(0.05, 0.08) if self._should_humanize() else 0.06)
        pyautogui.mouseUp()

        # Small pause after click
        time.sleep(self._random_delay() + 0.08)

        self._last_mouse_pos = (click_x, click_y)

    def dblclick(self, p: Point) -> None:
        """Humanized double-click"""
        self._thinking_pause()

        click_x, click_y = self._add_mouse_offset(p.x, p.y)
        self._humanized_move(click_x, click_y)

        time.sleep(self._random_delay() + 0.04)

        # First click
        pyautogui.mouseDown()
        time.sleep(random.uniform(0.05, 0.07) if self._should_humanize() else 0.06)
        pyautogui.mouseUp()

        # Pause between clicks (slightly randomized)
        time.sleep(random.uniform(0.04, 0.06) if self._should_humanize() else 0.05)

        # Second click
        pyautogui.mouseDown()
        time.sleep(random.uniform(0.04, 0.06) if self._should_humanize() else 0.05)
        pyautogui.mouseUp()

        time.sleep(self._random_delay() + 0.1)

        self._last_mouse_pos = (click_x, click_y)

    def sleep(self, seconds: float) -> None:
        """Sleep with optional small random variation"""
        if self._should_humanize() and seconds > 0.1:
            # Add up to 10% variation to longer sleeps
            variation = seconds * 0.1 * random.uniform(-1, 1)
            time.sleep(seconds + variation)
        else:
            time.sleep(seconds)

    def type_text(self, text: str) -> None:
        """Type with slightly randomized intervals"""
        interval = random.uniform(0.01, 0.02) if self._should_humanize() else 0.01
        pyautogui.typewrite(text, interval=interval)

    def press(self, key: str, presses: int = 1, interval: float = 0.01) -> None:
        if self._should_humanize():
            interval = random.uniform(0.01, 0.02)
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
            self.ctrl_combo("A");
            return
        if ks == ["ctrl", "c"]:
            self.ctrl_combo("C");
            return
        if ks == ["ctrl", "v"]:
            self.ctrl_combo("V");
            return
        pyautogui.hotkey(*ks)

    def paste_text(self, text: str) -> None:
        self.set_clipboard(text)
        self.ctrl_combo("V")

    def copy_hotkey(self) -> None:
        self.ctrl_combo("C")