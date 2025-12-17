from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class PurchaseLogger:
    """
    Лёгкий logger закупок:
      - держит записи в памяти (list[dict])
      - при каждой записи дописывает строку в CSV файл
      - печатает обновлённую таблицу в консоль (real-time)
    CSV columns (in Russian): timestamp,Название,Цена закупа,Награда,Профит,Общий профит
    Цена закупа — total spent на этот item (целое).
    Награда — выделенная часть общей награды (целое).
    Профит = Награда - Цена закупа.
    Общий профит — накопительный итог (сумма Profits).
    """
    CSV_HEADER = ["timestamp", "Название", "Цена закупа", "Награда", "Профит", "Общий профит"]

    def __init__(self, path: Path | str = "purchases_log.csv", append: bool = True) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.rows: List[Dict[str, str | int | float]] = []
        self.total_profit: int = 0

        # Ensure file exists and load existing cumulative profit (if append)
        if self.path.exists() and append:
            try:
                with self.path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        # Try to parse numeric fields conservatively
                        try:
                            profit = int(float(r.get("Профит", 0) or 0))
                        except Exception:
                            profit = 0
                        self.total_profit += profit
                        self.rows.append(r)
            except Exception:
                # If something wrong with existing file, continue with empty state
                self.rows = []
                self.total_profit = 0
        else:
            # create file with header
            with self.path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.CSV_HEADER)

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
        Внутри рассчитывается profit и cumulative total.
        Сразу записывает одну строчку в CSV и печатает таблицу.
        """
        ts = ts or datetime.utcnow()
        profit = int(reward) - int(price_spent)

        with self.lock:
            self.total_profit += profit
            row = {
                "timestamp": ts.isoformat(timespec="seconds"),
                "Название": name,
                "Цена закупа": int(price_spent),
                "Награда": int(reward),
                "Профит": int(profit),
                "Общий профит": int(self.total_profit),
            }
            self.rows.append(row)

            # append to CSV
            try:
                write_header = not self.path.exists()
                with self.path.open("a", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=self.CSV_HEADER)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception as e:
                # Do not raise — just log the problem via print (runtime logger will still have info)
                print(f"[PurchaseLogger] Failed to write CSV: {e}")

            # print table summary to console (real-time)
            self._print_realtime()

    def _print_realtime(self) -> None:
        # Print a compact table: last N rows (or all if small)
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
        print("=== PURCHASES (last {} rows) ===".format(len(last)))
        print(hdr)
        print(line)
        for r in last:
            print(sep.join(str(r.get(c, "")).ljust(widths[c]) for c in col_names))
        print(f"Accumulated profit: {self.total_profit}")
        print("=============================")
