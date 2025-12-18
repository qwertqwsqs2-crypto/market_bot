# gui/runtime_window.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import threading
import logging
from typing import Any
import queue
import time


class TkTextHandler(logging.Handler):
    """Logging handler that writes to a Tkinter Text widget (thread-safe via .after)."""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # schedule UI update on main thread
        self.text_widget.after(0, self._append, msg)

    def _append(self, msg: str) -> None:
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", msg + "\n")
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")


class RuntimeWindow(tk.Toplevel):
    """
    Runtime monitor window. Must be created from Tk main thread.
    Expects:
      - events_queue: queue.Queue producing dict-like events (type="purchase", item, spent, reward)
      - logger: root logger to attach text handler to
      - stop_callback: function to call when STOP button is pressed
      - ui_cleanup_callback: function to call for UI cleanup (OCR executor, etc.)
    """

    def __init__(self, master: tk.Misc, events_queue: "queue.Queue[dict]",
                 logger: logging.Logger, stop_callback: callable = None,
                 ui_cleanup_callback: callable = None):
        super().__init__(master)
        self.title("Runtime Monitor")
        self.geometry("700x1100")
        self.minsize(600, 700)
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.resizable(True, True)

        # Store callbacks
        self.stop_callback = stop_callback
        self.ui_cleanup_callback = ui_cleanup_callback

        # Modern dark theme colors (matching main window)
        self._bg = "#1a1a1a"
        self._bg_secondary = "#252525"
        self._bg_tertiary = "#2d2d2d"
        self._fg = "#e8e8e8"
        self._fg_secondary = "#b0b0b0"
        self._accent = "#4a9eff"
        self._success = "#4caf50"
        self._warning = "#ff9800"
        self._danger = "#f44336"

        self.configure(bg=self._bg)

        # Internal state
        self.events_queue = events_queue
        self._stop = False
        self._lock = threading.Lock()
        self._profit = 0
        self._paused = False

        self._create_widgets(logger)

        # Start polling
        self.after(200, self._poll_events)

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

    def _create_widgets(self, logger):
        """Create all UI widgets"""

        # Main container
        main_container = tk.Frame(self, bg=self._bg)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_container, bg=self._bg)
        header_frame.pack(fill="x", pady=(0, 20))

        title_label = tk.Label(
            header_frame,
            text="📊 Runtime Monitor",
            bg=self._bg,
            fg=self._fg,
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(side="left")

        subtitle_label = tk.Label(
            header_frame,
            text="Real-time bot activity and performance tracking",
            bg=self._bg,
            fg=self._fg_secondary,
            font=("Segoe UI", 10)
        )
        subtitle_label.pack(side="left", padx=(15, 0))

        # Two column layout using grid - VERTICAL layout
        columns_frame = tk.Frame(main_container, bg=self._bg, height=600)
        columns_frame.pack(fill="both", expand=True)
        columns_frame.pack_propagate(False)  # Prevent shrinking

        # Configure grid weights for vertical layout
        columns_frame.rowconfigure(0, weight=2)  # Table - smaller (40%)
        columns_frame.rowconfigure(1, weight=3)  # Log - bigger (60%)
        columns_frame.columnconfigure(0, weight=1)

        # ===== TOP - Purchases table =====
        top_section = tk.Frame(columns_frame, bg=self._bg)
        top_section.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        table_frame, table_content = self._create_label_frame(top_section, "💰 Purchase History")
        table_frame.pack(fill="both", expand=True)

        # Custom styled treeview
        tree_frame = tk.Frame(table_content, bg=self._bg_tertiary)
        tree_frame.pack(fill="both", expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(tree_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Treeview with custom style
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Runtime.Treeview",
                        background=self._bg_tertiary,
                        foreground=self._fg,
                        fieldbackground=self._bg_tertiary,
                        borderwidth=0,
                        font=("Segoe UI", 9))

        style.configure("Runtime.Treeview.Heading",
                        background=self._bg_secondary,
                        foreground=self._accent,
                        borderwidth=0,
                        font=("Segoe UI", 10, "bold"))

        style.map("Runtime.Treeview",
                  background=[("selected", self._accent)],
                  foreground=[("selected", "white")])

        columns = ("item", "spent", "reward", "profit")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            style="Runtime.Treeview",
            yscrollcommand=scrollbar.set
        )

        scrollbar.config(command=self.tree.yview)

        # Configure columns - wider for vertical layout
        column_widths = {"item": 300, "spent": 100, "reward": 100, "profit": 100}
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, anchor="center", width=column_widths.get(col, 70), minwidth=50)

        self.tree.pack(fill="both", expand=True)

        # ===== BOTTOM - Log =====
        bottom_section = tk.Frame(columns_frame, bg=self._bg)
        bottom_section.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        log_frame, log_content = self._create_label_frame(bottom_section, "📋 Activity Log")
        log_frame.pack(fill="both", expand=True)

        # Log text widget with scrollbar
        log_text_frame = tk.Frame(log_content, bg=self._bg_tertiary)
        log_text_frame.pack(fill="both", expand=True)

        log_scrollbar = tk.Scrollbar(log_text_frame, orient="vertical")
        log_scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(
            log_text_frame,
            state="disabled",
            bg=self._bg_tertiary,
            fg=self._fg,
            font=("Consolas", 9),
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
            yscrollcommand=log_scrollbar.set,
            wrap="none"  # No word wrap - horizontal scroll instead
        )
        self.log_text.pack(fill="both", expand=True)

        log_scrollbar.config(command=self.log_text.yview)

        # Configure log colors for different levels
        self.log_text.tag_config("INFO", foreground=self._fg)
        self.log_text.tag_config("WARNING", foreground=self._warning)
        self.log_text.tag_config("ERROR", foreground=self._danger)
        self.log_text.tag_config("DEBUG", foreground=self._fg_secondary)

        # Use helper function to add handler
        from logger import add_gui_handler
        self.log_handler = add_gui_handler(self.log_text, "market_bot")

        # ===== STATISTICS SECTION - Below log =====
        stats_frame = tk.Frame(main_container, bg=self._bg_secondary, relief="flat")
        stats_frame.pack(fill="x", pady=(20, 15))

        stats_inner = tk.Frame(stats_frame, bg=self._bg_secondary)
        stats_inner.pack(fill="x", padx=20, pady=15)

        # Session profit
        profit_container = tk.Frame(stats_inner, bg=self._bg_secondary)
        profit_container.pack(side="left")

        tk.Label(
            profit_container,
            text="💎 Session Profit:",
            bg=self._bg_secondary,
            fg=self._fg_secondary,
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=(0, 10))

        self.profit_var = tk.IntVar(value=0)
        self.profit_value_lbl = tk.Label(
            profit_container,
            textvariable=self.profit_var,
            bg=self._bg_secondary,
            fg=self._success,
            font=("Segoe UI", 16, "bold")
        )
        self.profit_value_lbl.pack(side="left")

        # Purchases count
        purchases_container = tk.Frame(stats_inner, bg=self._bg_secondary)
        purchases_container.pack(side="left", padx=(50, 0))

        tk.Label(
            purchases_container,
            text="🛒 Total Purchases:",
            bg=self._bg_secondary,
            fg=self._fg_secondary,
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=(0, 10))

        self.purchases_var = tk.IntVar(value=0)
        self.purchases_lbl = tk.Label(
            purchases_container,
            textvariable=self.purchases_var,
            bg=self._bg_secondary,
            fg=self._accent,
            font=("Segoe UI", 16, "bold")
        )
        self.purchases_lbl.pack(side="left")

        # ===== CONTROL BUTTONS =====
        control_frame = tk.Frame(main_container, bg=self._bg)
        control_frame.pack(fill="x", pady=(0, 0))

        self.stop_btn = self._create_modern_button(
            control_frame, "⏹ STOP BOT", self._on_stop, self._danger, width=12
        )
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.pause_btn = self._create_modern_button(
            control_frame, "⏸ Pause", self._toggle_pause, self._warning, width=10
        )
        self.pause_btn.pack(side="left", padx=(0, 10))

        clear_btn = self._create_modern_button(
            control_frame, "🗑️ Clear History", self._clear_history, self._bg_tertiary, width=12
        )
        clear_btn.pack(side="left")

    def _on_close(self) -> None:
        # When user closes window, treat it as STOP
        if self.stop_callback:
            self.stop_callback()
        # ДОБАВЛЕНО: Cleanup UI resources
        if self.ui_cleanup_callback:
            try:
                self.ui_cleanup_callback()
            except Exception as e:
                import logging
                logging.getLogger("market_bot").warning("UI cleanup on close failed: %s", e)
        self.destroy()

    def _on_stop(self) -> None:
        """Handle STOP button press"""
        if self.stop_callback:
            self.stop_callback()
        # Window will be destroyed by main window after bot stops

    def destroy(self) -> None:
        # remove handler when destroyed
        try:
            root_logger = logging.getLogger("market_bot")
            root_logger.removeHandler(self.log_handler)
        except Exception:
            pass
        super().destroy()

    def _toggle_pause(self) -> None:
        # The actual pause/resume must be handled by main control logic via pause_event.
        # Here we just toggle label for convenience (actual state lives in main).
        self._paused = not self._paused
        if self._paused:
            self.pause_btn.config(text="▶ Resume")
        else:
            self.pause_btn.config(text="⏸ Pause")

    def _clear_history(self) -> None:
        """Clear purchase history table"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._profit = 0
        self.profit_var.set(0)
        self.purchases_var.set(0)

    def _poll_events(self) -> None:
        """Pulls events_queue and updates UI (called periodically via after)."""
        try:
            while True:
                ev = self.events_queue.get_nowait()
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") == "purchase":
                    item = ev.get("item", "Unknown")
                    spent = int(ev.get("spent") or 0)
                    reward = int(ev.get("reward") or 0)
                    profit = reward - spent
                    self._profit += profit
                    self.profit_var.set(self._profit)

                    # Update purchases count
                    current_count = self.purchases_var.get()
                    self.purchases_var.set(current_count + 1)

                    # Insert with color coding based on profit
                    item_id = self.tree.insert("", 0, values=(item, spent, reward, profit))

                    # Color code profit column
                    if profit > 0:
                        self.tree.item(item_id, tags=("profit",))
                    elif profit < 0:
                        self.tree.item(item_id, tags=("loss",))

                # other event types can be handled here
        except queue.Empty:
            pass
        if not self._stop:
            self.after(200, self._poll_events)

    def stop(self) -> None:
        with self._lock:
            self._stop = True