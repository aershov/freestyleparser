"""Диагностика: почему попытка 0004 стартовала на 146с, а не раньше.

Прогоняет YOLO по кадрам вокруг заданного времени и печатает детекции
персон/лодок с rel (доля bbox внутри ROI-прямоугольника).
"""
import sys
import numpy as np
import cv2
from ultralytics import YOLO

import sys as _sys
SRC = _sys.argv[3] if len(_sys.argv) > 3 else '/Volumes/Untitled/DCIM/999_PANA/P9990761.MP4'
SLOWMO = 4
SCALE = 0.4
ROI_POLY = [(23.464912280701753, 14.166666666666666), (46.271929824561404, 3.75),
            (68.42105263157895, 6.25), (75.43859649122807, 9.583333333333332),
            (97.14912280701755, 30.833333333333332), (97.36842105263158, 97.5),
            (16.885964912280702, 93.75), (7.2368421052631575, 67.08333333333333),
            (7.675438596491228, 43.75)]
MAIN_MIN_REL = 0.5

xs = [p[0] for p in ROI_POLY]
ys = [p[1] for p in ROI_POLY]
L, T, R, B = min(xs), min(ys), max(xs), max(ys)

model = YOLO('models/yolo11s-seg.pt')
src = cv2.VideoCapture(SRC)
fps_c = src.get(cv2.CAP_PROP_FPS)
W = src.get(cv2.CAP_PROP_FRAME_WIDTH)
H = src.get(cv2.CAP_PROP_FRAME_HEIGHT)
Lpx, Tpx, Rpx, Bpx = [v / 100 * (W if i % 2 == 0 else H) for i, v in enumerate([L, T, R, B])]

t0, t1 = float(sys.argv[1]), float(sys.argv[2])
step = 0.25  # сек реального времени между пробами

t = t0
while t <= t1:
    src.set(cv2.CAP_PROP_POS_MSEC, t * SLOWMO * 1000)
    ok, frame = src.read()
    if not ok:
        break
    roi_crop = frame[int(Tpx):int(Bpx), int(Lpx):int(Rpx)]
    small = cv2.resize(roi_crop, None, fx=SCALE, fy=SCALE)
    res = model(small, verbose=False)[0]
    out = []
    for box in res.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        if conf < 0.3:
            continue
        x1, y1, x2, y2 = [c / SCALE for c in box.xyxy[0].cpu().numpy()]
        fx1, fy1 = x1 + Lpx, y1 + Tpx
        fx2, fy2 = x2 + Lpx, y2 + Tpx
        iw = max(min(fx2, Rpx) - max(fx1, Lpx), 0.0)
        ih = max(min(fy2, Bpx) - max(fy1, Tpx), 0.0)
        area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
        rel = iw * ih / area
        in_roi = 'P' if (name == 'person' and rel >= MAIN_MIN_REL) else ' '
        out.append(f"{name[:4]} c={conf:.2f} rel={rel:.2f}{in_roi} "
                   f"x=({fx1 / W * 100:.0f},{fy1 / H * 100:.0f})-({fx2 / W * 100:.0f},{fy2 / H * 100:.0f})%")
    print(f"t={t:6.2f} file={t * SLOWMO:6.1f} | " + ('; '.join(out) if out else 'нет детекций'))
    t += step
src.release()
