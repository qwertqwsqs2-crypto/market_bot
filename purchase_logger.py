from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class PurchaseLogger:
    """
    Лёгкий logger закупок:
      - держит записи текущей сессии в памяти (list[dict])
      - при каждой записи дописывает строку в CSV файл
      - печатает обновлённую таблицу в консоль (real-time)
    CSV columns (in Russian): timestamp,Название,Цена закупа,Награда,Профит,Общий профит
    Цена закупа — total spent на этот item (целое).
    Награда — выделенная часть общей награды (целое).
    Профит = Награда - Цена закупа.
    Общий профит — накопительный итог ТОЛЬКО ДЛЯ ТЕКУЩЕЙ СЕССИИ.
    """
    CSV_HEADER = ["timestamp", "Название", "Цена закупа", "Награда", "Профит", "Общий профит"]

    def __init__(self, path: Path | str = "purchases_log.csv", append: bool = True) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.rows: List[Dict[str, str | int | float]] = []  # Only current session
        self.total_profit: int = 0  # Only current session

        # Ensure file exists with header (append mode)
        if not self.path.exists():
            # Create file with header if it doesn't exist
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.CSV_HEADER)
        elif not append:
            # If append=False, overwrite file with just header
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.CSV_HEADER)

        # Note: We don't load old data - session profit starts at 0

    def record(
        self,
        name: str,
        price_spent: int,
        reward: int,
        *,
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Добавить запись: name, total spent (int), reward (int).
        Внутри рассчитывается profit и cumulative total ДЛЯ ТЕКУЩЕЙ СЕССИИ.
        Сразу записывает одну строчку в CSV и печатает таблицу.
        """
        ts = ts or datetime.utcnow()
        profit = int(reward) - int(price_spent)

        with self.lock:
            self.total_profit += profit  # Session profit
            row = {
                "timestamp": ts.isoformat(timespec="seconds"),
                "Название": name,
                "Цена закупа": int(price_spent),
                "Награда": int(reward),
                "Профит": int(profit),
                "Общий профит": int(self.total_profit),  # Session cumulative
            }
            self.rows.append(row)

            # append to CSV
            try:
                with self.path.open("a", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=self.CSV_HEADER)
                    writer.writerow(row)
            except Exception as e:
                # Do not raise — just log the problem via print (runtime logger will still have info)
                print(f"[PurchaseLogger] Failed to write CSV: {e}")

            # print table summary to console (real-time)
            self._print_realtime()

    def _print_realtime(self) -> None:
        """Print a compact table: last N rows (or all if small)"""
        last = self.rows[-10:]
        col_names = ["Название", "Цена закупа", "Награда", "Профит", "Общий профит"]
        widths = {c: len(c) for c in col_names}
        for r in last:
            for c in col_names:
                widths[c] = max(widths[c], len(str(r.get(c, ""))))

        # header
        sep = " | "
        hdr = sep.join(c.ljust(widths[c]) for c in col_names)
        line = "-".join("-" * (widths[c] + 2) for c in col_names)
        print()
        print("=== PURCHASES (last {} rows) - SESSION ONLY ===".format(len(last)))
        print(hdr)
        print(line)
        for r in last:
            print(sep.join(str(r.get(c, "")).ljust(widths[c]) for c in col_names))
        print(f"Session profit: {self.total_profit}")
        print("=" * 50)