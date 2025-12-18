# gui/main_window.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import json
from pathlib import Path
import logging
import time

from gui.runtime_window import RuntimeWindow


class ModernCheckbutton(tk.Frame):
    """Custom styled checkbutton"""

    def __init__(self, parent, text, variable, **kwargs):
        super().__init__(parent, bg="#2d2d2d", **kwargs)
        self.variable = variable

        self.check = tk.Checkbutton(
            self,
            text=text,
            variable=variable,
            bg="#2d2d2d",
            fg="#e8e8e8",
            selectcolor="#252525",
            activebackground="#2d2d2d",
            activeforeground="#4a9eff",
            font=("Segoe UI", 9),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=10,
            pady=8
        )
        self.check.pack(fill="both", expand=True)


class MainWindow(tk.Tk):
    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        self.title("Market Bot Control Panel")
        self.geometry("1200x700")
        self.minsize(1000, 600)
        self.resizable(True, True)
        self.config_path = Path(config_path)

        # Modern dark theme colors
        self._bg = "#1a1a1a"
        self._bg_secondary = "#252525"
        self._bg_tertiary = "#2d2d2d"
        self._fg = "#e8e8e8"
        self._fg_secondary = "#b0b0b0"
        self._accent = "#4a9eff"
        self._accent_hover = "#5cadff"
        self._success = "#4caf50"
        self._warning = "#ff9800"
        self._danger = "#f44336"

        self.configure(bg=self._bg)

        # Logging - setup once
        from logger import setup_logging
        setup_logging(level=logging.INFO)
        self.logger = logging.getLogger("market_bot")

        # Load config (raw and typed)
        try:
            from config import AppConfig
            self.cfg = AppConfig.load(self.config_path)
            self.raw_cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Config error", f"Cannot load config: {e}")
            raise

        # Load quests
        from quests_db import ALL_QUESTS
        from bot import parse_quests_db
        self.quests = parse_quests_db(ALL_QUESTS)
        self.quest_names = [q.quest for q in self.quests]

        # Quest checkbox variables
        self.quest_vars = [tk.BooleanVar(value=True) for _ in self.quest_names]

        self._create_widgets()
        self._bot_thread = None
        self._stop_event = None
        self._pause_event = None
        self._events_queue = None
        self.runtime_win = None

        # Bind resize event for responsive layout
        self.bind("<Configure>", self._on_window_resize)
        self._last_width = self.winfo_width()

    def _create_modern_button(self, parent, text, command, bg_color, width=None):
        """Create a modern flat button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg="white",
            activebackground=bg_color,
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=10,
            cursor="hand2",
            width=width or 12
        )

        # Hover effect
        def on_enter(e):
            btn.config(bg=self._lighten_color(bg_color))

        def on_leave(e):
            btn.config(bg=bg_color)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        return btn

    def _lighten_color(self, hex_color):
        """Lighten a hex color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r = min(255, int(r * 1.15))
        g = min(255, int(g * 1.15))
        b = min(255, int(b * 1.15))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _create_label_frame(self, parent, text):
        """Create a modern label frame"""
        frame = tk.Frame(parent, bg=self._bg_secondary, relief="flat")

        # Header with icon and title
        header = tk.Frame(frame, bg=self._bg_secondary)
        header.pack(fill="x", padx=15, pady=(15, 10))

        label = tk.Label(
            header,
            text=text,
            bg=self._bg_secondary,
            fg=self._accent,
            font=("Segoe UI", 11, "bold")
        )
        label.pack(side="left")

        # Separator line
        sep = tk.Frame(frame, height=1, bg="#3a3a3a")
        sep.pack(fill="x", padx=15)

        # Content frame
        content = tk.Frame(frame, bg=self._bg_secondary)
        content.pack(fill="both", expand=True, padx=15, pady=15)

        return frame, content

    def _on_window_resize(self, event):
        """Handle window resize for responsive layout"""
        if event.widget == self and hasattr(self, '_quests_container'):
            current_width = event.width
            if abs(current_width - self._last_width) > 50:
                self._last_width = current_width
                self._reflow_quest_checkboxes()

    def _reflow_quest_checkboxes(self):
        """Reflow quest checkboxes based on available width"""
        if not hasattr(self, '_quests_container'):
            return

        # Clear existing checkboxes
        for widget in self._quests_container.winfo_children():
            widget.destroy()

        # Always use 2 columns for better readability
        columns = 2

        # Create grid of checkboxes
        for idx, (quest_name, var) in enumerate(zip(self.quest_names, self.quest_vars)):
            row = idx // columns
            col = idx % columns

            cb = ModernCheckbutton(self._quests_container, quest_name, var)
            cb.grid(row=row, column=col, sticky="w", padx=3, pady=3)

        # Configure column weights for even distribution
        for col in range(columns):
            self._quests_container.columnconfigure(col, weight=1)

    def _create_widgets(self):
        """Create all UI widgets with horizontal layout"""

        # Main container
        main_container = tk.Frame(self, bg=self._bg)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_container, bg=self._bg)
        header_frame.pack(fill="x", pady=(0, 20))

        title_label = tk.Label(
            header_frame,
            text="⚙️ Market Bot Control Panel",
            bg=self._bg,
            fg=self._fg,
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(side="left")

        subtitle_label = tk.Label(
            header_frame,
            text="Configure and monitor your automated market bot",
            bg=self._bg,
            fg=self._fg_secondary,
            font=("Segoe UI", 10)
        )
        subtitle_label.pack(side="left", padx=(15, 0))

        # Two column layout
        columns_frame = tk.Frame(main_container, bg=self._bg)
        columns_frame.pack(fill="both", expand=True)

        # LEFT COLUMN - Configuration
        left_column = tk.Frame(columns_frame, bg=self._bg)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Configuration section
        config_frame, config_content = self._create_label_frame(left_column, "⚙️ Configuration")
        config_frame.pack(fill="x", pady=(0, 15))

        config_grid = tk.Frame(config_content, bg=self._bg_secondary)
        config_grid.pack(fill="x")
        config_grid.columnconfigure(0, weight=1)
        config_grid.columnconfigure(1, weight=1)

        # Level selection
        level_frame = tk.Frame(config_grid, bg=self._bg_secondary)
        level_frame.grid(row=0, column=0, sticky="ew", padx=(0, 10), pady=5)

        tk.Label(level_frame, text="Level:", bg=self._bg_secondary, fg=self._fg,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

        level_buttons = tk.Frame(level_frame, bg=self._bg_secondary)
        level_buttons.pack(anchor="w")

        self.level_var = tk.IntVar(value=60)
        for val in [60, 65]:
            rb = tk.Radiobutton(
                level_buttons,
                text=f"Level {val}",
                variable=self.level_var,
                value=val,
                bg=self._bg_secondary,
                fg=self._fg,
                selectcolor=self._bg_tertiary,
                activebackground=self._bg_secondary,
                activeforeground=self._accent,
                font=("Segoe UI", 9),
                relief="flat"
            )
            rb.pack(side="left", padx=(0, 15))

        # Sets
        sets_frame = tk.Frame(config_grid, bg=self._bg_secondary)
        sets_frame.grid(row=0, column=1, sticky="ew", pady=5)

        tk.Label(sets_frame, text="Number of Sets:", bg=self._bg_secondary, fg=self._fg,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

        self.sets_var = tk.IntVar(value=1)
        sets_spinbox = tk.Spinbox(
            sets_frame,
            from_=1, to=999,
            textvariable=self.sets_var,
            width=10,
            bg=self._bg_tertiary,
            fg=self._fg,
            buttonbackground=self._bg_tertiary,
            relief="flat",
            font=("Segoe UI", 10)
        )
        sets_spinbox.pack(anchor="w")

        # Profit threshold
        profit_frame, profit_content = self._create_label_frame(left_column, "💰 Profit Threshold")
        profit_frame.pack(fill="x", pady=(0, 15))

        self.profit_enabled = tk.BooleanVar(value=self.cfg.profit.enabled)
        enable_cb = tk.Checkbutton(
            profit_content,
            text="Enable profit threshold filtering",
            variable=self.profit_enabled,
            bg=self._bg_secondary,
            fg=self._fg,
            selectcolor=self._bg_tertiary,
            activebackground=self._bg_secondary,
            activeforeground=self._accent,
            font=("Segoe UI", 9),
            relief="flat"
        )
        enable_cb.pack(anchor="w", pady=(0, 10))

        # Profit settings grid
        profit_grid = tk.Frame(profit_content, bg=self._bg_secondary)
        profit_grid.pack(fill="x")
        profit_grid.columnconfigure(1, weight=1)

        # Mode
        mode_frame = tk.Frame(profit_grid, bg=self._bg_secondary)
        mode_frame.grid(row=0, column=0, sticky="nw", padx=(0, 15))

        tk.Label(mode_frame, text="Mode:", bg=self._bg_secondary, fg=self._fg,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))

        self.profit_mode = tk.StringVar(value=self.cfg.profit.mode or "percent")
        for mode_val, mode_text in [("percent", "Percentage"), ("absolute", "Absolute")]:
            rb = tk.Radiobutton(
                mode_frame,
                text=mode_text,
                variable=self.profit_mode,
                value=mode_val,
                bg=self._bg_secondary,
                fg=self._fg,
                selectcolor=self._bg_tertiary,
                activebackground=self._bg_secondary,
                activeforeground=self._accent,
                font=("Segoe UI", 9),
                relief="flat"
            )
            rb.pack(anchor="w", pady=2)

        # Values
        values_frame = tk.Frame(profit_grid, bg=self._bg_secondary)
        values_frame.grid(row=0, column=1, sticky="ew")

        # Percent
        percent_row = tk.Frame(values_frame, bg=self._bg_secondary)
        percent_row.pack(fill="x", pady=(0, 8))
        tk.Label(percent_row, text="Min Percent (0-1):", bg=self._bg_secondary,
                 fg=self._fg, font=("Segoe UI", 9), width=18).pack(side="left")
        self.profit_percent = tk.DoubleVar(value=self.cfg.profit.min_profit_percent)
        percent_entry = tk.Entry(
            percent_row,
            textvariable=self.profit_percent,
            bg=self._bg_tertiary,
            fg=self._fg,
            relief="flat",
            font=("Segoe UI", 10),
            insertbackground=self._fg
        )
        percent_entry.pack(side="left", fill="x", expand=True, padx=(10, 0), ipady=4)

        # Absolute
        absolute_row = tk.Frame(values_frame, bg=self._bg_secondary)
        absolute_row.pack(fill="x")
        tk.Label(absolute_row, text="Min Absolute:", bg=self._bg_secondary,
                 fg=self._fg, font=("Segoe UI", 9), width=18).pack(side="left")
        self.profit_abs = tk.IntVar(value=self.cfg.profit.min_profit_absolute)
        absolute_entry = tk.Entry(
            absolute_row,
            textvariable=self.profit_abs,
            bg=self._bg_tertiary,
            fg=self._fg,
            relief="flat",
            font=("Segoe UI", 10),
            insertbackground=self._fg
        )
        absolute_entry.pack(side="left", fill="x", expand=True, padx=(10, 0), ipady=4)

        # Runtime options
        runtime_frame, runtime_content = self._create_label_frame(left_column, "🔧 Runtime Options")
        runtime_frame.pack(fill="both", expand=True)

        self.simulate_ui = tk.BooleanVar(value=self.cfg.runtime.simulate_ui_actions)
        simulate_cb = tk.Checkbutton(
            runtime_content,
            text="🧪 Simulate UI actions (testing mode)",
            variable=self.simulate_ui,
            bg=self._bg_secondary,
            fg=self._fg,
            selectcolor=self._bg_tertiary,
            activebackground=self._bg_secondary,
            activeforeground=self._accent,
            font=("Segoe UI", 9),
            relief="flat"
        )
        simulate_cb.pack(anchor="w")

        # RIGHT COLUMN - Quest Selection
        right_column = tk.Frame(columns_frame, bg=self._bg)
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Quests section
        quests_frame, quests_content = self._create_label_frame(right_column, "📋 Quest Selection")
        quests_frame.pack(fill="both", expand=True)

        # Selection buttons
        selection_btns = tk.Frame(quests_content, bg=self._bg_secondary)
        selection_btns.pack(fill="x", pady=(0, 10))

        btn_select = self._create_modern_button(
            selection_btns, "✓ Select All", self._select_all, self._success, width=10
        )
        btn_select.pack(side="left", padx=(0, 10))

        btn_clear = self._create_modern_button(
            selection_btns, "✗ Clear All", self._clear_selection, self._danger, width=10
        )
        btn_clear.pack(side="left")

        self._selected_label = tk.Label(
            selection_btns,
            text="",
            bg=self._bg_secondary,
            fg=self._accent,
            font=("Segoe UI", 10, "bold")
        )
        self._selected_label.pack(side="right")

        # Update count
        def update_count(*args):
            count = sum(1 for var in self.quest_vars if var.get())
            total = len(self.quest_vars)
            self._selected_label.config(text=f"Selected: {count}/{total}")

        for var in self.quest_vars:
            var.trace_add("write", update_count)
        update_count()

        # Scrollable quests
        quest_canvas = tk.Canvas(quests_content, bg=self._bg_tertiary,
                                 highlightthickness=0)
        quest_scrollbar = tk.Scrollbar(quests_content, orient="vertical",
                                       command=quest_canvas.yview)

        self._quests_container = tk.Frame(quest_canvas, bg=self._bg_tertiary)
        self._quests_container.bind(
            "<Configure>",
            lambda e: quest_canvas.configure(scrollregion=quest_canvas.bbox("all"))
        )

        quest_canvas.create_window((0, 0), window=self._quests_container, anchor="nw")
        quest_canvas.configure(yscrollcommand=quest_scrollbar.set)

        quest_canvas.pack(side="left", fill="both", expand=True)
        quest_scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            quest_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        quest_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._reflow_quest_checkboxes()

        # BOTTOM - Control buttons and status
        bottom_frame = tk.Frame(main_container, bg=self._bg)
        bottom_frame.pack(fill="x", pady=(20, 0))

        # Control buttons
        control_frame = tk.Frame(bottom_frame, bg=self._bg)
        control_frame.pack(fill="x", pady=(0, 15))

        left_btns = tk.Frame(control_frame, bg=self._bg)
        left_btns.pack(side="left")

        self.start_btn = self._create_modern_button(
            left_btns, "▶ START BOT", self._on_start, self._success, width=12
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.pause_btn = self._create_modern_button(
            left_btns, "⏸ PAUSE", self._on_pause, self._warning, width=10
        )
        self.pause_btn.config(state="disabled")
        self.pause_btn.pack(side="left")

        self.save_btn = self._create_modern_button(
            control_frame, "💾 Save Settings", self._save_settings, self._accent, width=14
        )
        self.save_btn.pack(side="right")

        # Status bar
        status_frame = tk.Frame(bottom_frame, bg=self._bg_tertiary, relief="flat")
        status_frame.pack(fill="x")

        status_inner = tk.Frame(status_frame, bg=self._bg_tertiary)
        status_inner.pack(fill="x", padx=15, pady=12)

        tk.Label(
            status_inner,
            text="Status:",
            bg=self._bg_tertiary,
            fg=self._fg_secondary,
            font=("Segoe UI", 10, "bold")
        ).pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(
            status_inner,
            textvariable=self.status_var,
            bg=self._bg_tertiary,
            fg=self._accent,
            font=("Segoe UI", 10, "bold")
        )
        self.status_label.pack(side="left")

    def _select_all(self):
        for var in self.quest_vars:
            var.set(True)

    def _clear_selection(self):
        for var in self.quest_vars:
            var.set(False)

    def _save_settings(self):
        try:
            cfg_path = self.config_path
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            raw_profit = raw.get("profit", {})
            raw_profit["enabled"] = bool(self.profit_enabled.get())
            raw_profit["mode"] = str(self.profit_mode.get())
            raw_profit["min_profit_percent"] = float(self.profit_percent.get())
            raw_profit["min_profit_absolute"] = int(self.profit_abs.get())
            raw["profit"] = raw_profit

            runtime = raw.get("runtime", {})
            runtime["simulate_ui_actions"] = bool(self.simulate_ui.get())
            raw["runtime"] = runtime

            cfg_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
            messagebox.showinfo("Success", f"Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def _on_start(self):
        selected_indices = [i for i, var in enumerate(self.quest_vars) if var.get()]

        if not selected_indices:
            messagebox.showwarning("No Quests Selected",
                                   "Please select at least one quest to start the bot.")
            return

        selected_quests = [self.quests[i] for i in selected_indices]
        level = int(self.level_var.get())
        sets = int(self.sets_var.get())
        mode_str = "simulate" if self.simulate_ui.get() else "run"

        try:
            from config import AppConfig
            from input_controller import GameInput, SimulatedInput
            from ocr_utils import OCRReader
            from ui import MarketUI
            from bot import MarketBot, BotContext
            from models import Mode
        except Exception as e:
            messagebox.showerror("Import Error", f"Cannot start bot: {e}")
            return

        cfg = self.cfg

        if (mode_str == "simulate") and (not cfg.runtime.simulate_ui_actions):
            inp = SimulatedInput(logger=self.logger)
        else:
            if cfg.runtime.input_backend != "pyautogui":
                self.logger.warning("Unknown input_backend=%r; fallback to pyautogui.",
                                    cfg.runtime.input_backend)
            inp = GameInput(wait_between_clicks=cfg.timing.wait_between_clicks,
                            logger=self.logger)

        ocr = OCRReader(
            use_easyocr=cfg.ocr.use_easyocr,
            langs=cfg.ocr.easyocr_langs,
            use_tesseract=cfg.ocr.use_tesseract_fallback,
            min_conf=cfg.ocr.min_confidence,
            prefer_gpu=cfg.ocr.easyocr_gpu,
        )

        ui = MarketUI(cfg=cfg, inp=inp, ocr=ocr, logger=self.logger)

        events_q = queue.Queue()
        stop_ev = threading.Event()
        pause_ev = threading.Event()

        try:
            mode_enum = Mode(mode_str)
        except Exception:
            mode_enum = Mode(mode_str.lower())

        ctx = BotContext(cfg=cfg, ui=ui, logger=self.logger, mode=mode_enum,
                         level=level, sets=sets,
                         stop_event=stop_ev, pause_event=pause_ev, events=events_q)

        bot = MarketBot(ctx)

        def _run_bot():
            try:
                self.status_var.set("🟢 Running")
                self.status_label.configure(fg=self._success)
                bot.run(selected_quests)
            except Exception as e:
                self.logger.exception("Bot thread error: %s", e)
            finally:
                self.after(0, lambda: self._on_thread_finished())

        self._stop_event = stop_ev
        self._pause_event = pause_ev
        self._events_queue = events_q
        self._bot_thread = threading.Thread(target=_run_bot, daemon=True)
        self._bot_thread.start()

        # Create runtime window with stop callback
        self.runtime_win = RuntimeWindow(
            self,
            events_queue=events_q,
            logger=self.logger,
            stop_callback=self._on_stop
        )

        # Minimize main window when bot starts
        self.iconify()

        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal")

    def _on_thread_finished(self):
        # Restore main window when bot stops
        self.deiconify()
        self.lift()

        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.status_var.set("⚪ Stopped")
        self.status_label.configure(fg=self._fg_secondary)
        if self.runtime_win:
            try:
                self.runtime_win.destroy()  # Destroy instead of stop
                self.runtime_win = None
            except Exception:
                pass

    def _on_stop(self):
        """Called when STOP button is pressed (from Runtime window)"""
        if self._stop_event:
            self._stop_event.set()
            self.status_var.set("🟡 Stopping...")
            self.status_label.configure(fg=self._warning)
            # Disable pause while stopping
            if hasattr(self, 'pause_btn'):
                self.pause_btn.config(state="disabled")
        else:
            self.status_var.set("⚪ Idle")
            self.status_label.configure(fg=self._fg_secondary)

    def _on_pause(self):
        if not self._pause_event:
            return
        if not self._pause_event.is_set():
            self._pause_event.set()
            self.pause_btn.config(text="▶ RESUME")
            self.status_var.set("🟡 Paused")
            self.status_label.configure(fg=self._warning)
        else:
            self._pause_event.clear()
            self.pause_btn.config(text="⏸ PAUSE")
            self.status_var.set("🟢 Running")
            self.status_label.configure(fg=self._success)