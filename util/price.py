import time
import json
import os
import re
import argparse

import numpy as np
import cv2
import mss
import easyocr

ROI_FILE = "roi.json"

def format_price(d: str) -> str:
    # "3000" -> "3 000", "2948" -> "2 948"
    d = re.sub(r"\D+", "", d or "")
    if not d:
        return ""
    parts = []
    while d:
        parts.append(d[-3:])
        d = d[:-3]
    return " ".join(reversed(parts))

def load_roi():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, "r", encoding="utf-8") as f:
            r = json.load(f)
        return int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    return None

def save_roi(x, y, w, h):
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, f, ensure_ascii=False, indent=2)

def pick_roi(monitor_index: int):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]  # {"left","top","width","height"}
        shot = sct.grab(mon)
        img = np.array(shot)[:, :, :3]  # BGRA -> BGR

    # selectROI работает в координатах этого снимка (относительно монитора)
    r = cv2.selectROI("Select ROI (ENTER confirm, ESC cancel)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        print("[CANCEL] ROI not saved.")
        return

    # конвертируем в "глобальные" координаты экрана
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        gx = mon["left"] + x
        gy = mon["top"] + y

    save_roi(gx, gy, w, h)
    print(f"[OK] Saved ROI to {ROI_FILE}: x={gx}, y={gy}, w={w}, h={h}")

def preprocess_for_ocr(bgr_roi, ocr_scale: float):
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    if ocr_scale != 1.0:
        gray = cv2.resize(gray, None, fx=ocr_scale, fy=ocr_scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE + лёгкая резкость, чтобы 3 не превращалась в 8 из-за "залепания" дырок
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return sharp

def draw_label(img, x, y, text):
    # тень + текст
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true", help="Выбрать ROI и сохранить в roi.json")
    ap.add_argument("--monitor", type=int, default=1, help="Номер монитора для выбора ROI (mss monitors index), обычно 1")
    ap.add_argument("--gpu", action="store_true", help="Использовать GPU (если torch+cudа настроены)")
    ap.add_argument("--ocr-scale", type=float, default=3.0, help="Внешний апскейл перед OCR (1.0..4.0)")
    ap.add_argument("--update", type=float, default=0.25, help="Пауза между распознаваниями (сек)")
    ap.add_argument("--minconf", type=float, default=0.40, help="Минимальная уверенность для отображения")
    args = ap.parse_args()

    if args.pick:
        pick_roi(args.monitor)
        return

    roi = load_roi()
    if not roi:
        print("ROI не задан. Запусти: python market_ocr_debug.py --pick")
        return

    x, y, w, h = roi
    print(f"[ROI] x={x}, y={y}, w={w}, h={h}")
    print("[INFO] Лучше работает в borderless/windowed. Выход: ESC (если окно в фокусе) или Ctrl+C.")

    reader = easyocr.Reader(["en"], gpu=args.gpu)

    # настройки EasyOCR под цифры
    ocr_kwargs = dict(
        allowlist="0123456789",
        decoder="beamsearch",
        beamWidth=10,
        detail=1,
        paragraph=False,
        min_size=8,
        # mag_ratio оставляем 1.0, потому что мы уже апскейлим сами
        mag_ratio=1.0,
        canvas_size=2560,
        contrast_ths=0.2,
        adjust_contrast=0.8,
        text_threshold=0.6,
        low_text=0.3,
        link_threshold=0.3,
    )

    win = "Market OCR Debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(win, 900, 700)

    with mss.mss() as sct:
        while True:
            t0 = time.time()

            shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            bgr = np.array(shot)[:, :, :3]  # BGRA -> BGR

            proc = preprocess_for_ocr(bgr, args.ocr_scale)

            # цветная картинка того же размера для рисования боксов
            if args.ocr_scale != 1.0:
                vis = cv2.resize(bgr, None, fx=args.ocr_scale, fy=args.ocr_scale, interpolation=cv2.INTER_NEAREST)
            else:
                vis = bgr.copy()

            res = reader.readtext(proc, **ocr_kwargs)

            # Рисуем боксы и подписи около каждой цены
            for bbox, text, conf in res:
                digits = re.sub(r"\D+", "", text or "")
                if not digits or float(conf) < args.minconf:
                    continue

                # bbox: 4 точки [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

                # ставим подпись рядом с верхним левым углом
                x0, y0 = int(pts[:,0].min()), int(pts[:,1].min())
                label = f"{format_price(digits)} ({float(conf):.2f})"
                draw_label(vis, x0, max(20, y0 - 8), label)

            dt_ms = int((time.time() - t0) * 1000)
            draw_label(vis, 10, 25, f"{dt_ms}ms  update={args.update:.2f}s  minconf={args.minconf:.2f}")

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            time.sleep(args.update)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
