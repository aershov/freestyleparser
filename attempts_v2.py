"""attempts_v2: попытки по 'главному атлету' + вектор свойств атлета.

Логика разделения:
- Главный атлет кадра = персон с максимальной долей bbox внутри ROI
  (залезающие частично атлеты проигрывают).
- Попытка живёт, пока главный атлет в ROI (с непрерывностью трека:
  при кратковременной потере трека берём лучшего кандидата).
- Пауза >= MIN_PAUSE без главного атлета закрывает попытку.

Вектор свойств attempts (каждый атрибут опционален, копится только из
качественных наблюдений):
- vest:   HSV-гистограмма туловища (по seg-маске)
- helmet: HSV-гистограмма головы (по seg-маске)
- boat:   накопленный профиль оттенка ближайших чистых детекций лодки
- dino:   усреднённый DINOv2-эмбеддинг маскированного кропа
"""
import argparse
import json
import os
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

ROI_PERCENTAGES = (0, 23, 95, 90)
YOLO_CONFIDENCE_THRESHOLD = 0.4
BOAT_CLASSES = ('boat', 'surfboard')

# Параметры попыток
ANALYSIS_FPS = 5.0           # сэмплов анализа на реальную секунду (как в splitter.py)
MIN_ATTEMPT_DURATION = 3
MIN_PAUSE_DURATION = 1.5     # отсутствие главного (по presence-ratio) для закрытия
ATTEMPT_START_PADDING = 2
ATTEMPT_END_PADDING = 1
MAIN_MIN_REL = 0.5       # минимальная доля bbox в ROI, чтобы считаться главным
BANK_SIZE = 15
MIN_QUALITY = 0.12
VEST_DIM = 1152   # HSV-гистограмма жилета/шлема (18x8x8, sqrt)
REID_DIM = 512    # ансамбль OSNet+AIN (усреднённый, L2)
BOAT_MIN_FRAC = 0.30
MERGE_GAP_MAX = 8.0        # макс пауза между попытками для склейки, с
MERGE_BOAT_SIM = 0.7       # порог похожести цвета лодки для склейки
FALLBACK_PRESENCE_CAP = 3.0  # макс длительность presence по лодке (атлет под водой), с


def default_device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', default='temp/attempts_v2')
    p.add_argument('--scale', type=float, default=0.4)
    p.add_argument('--roi', type=float, nargs=4, default=ROI_PERCENTAGES)
    p.add_argument('--no-embed', action='store_true')
    p.add_argument('--trace', action='store_true', help='Печать выбора главного атлета')
    return p.parse_args()


def hsv_hist(crop, mask=None):
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = (mask > 0.5).astype(np.uint8) if mask is not None else None
    hist = cv2.calcHist([hsv], [0, 1, 2], m, [18, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    if hist.sum() < 20:
        return None
    return np.sqrt(hist / hist.sum())


def hue_bins(crop):
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])[:, 1:].sum(axis=1)
    if hist.sum() < 10:
        return None
    return hist / hist.sum(), float(hsv[..., 1].mean()) / 255.0


class Embedder:
    """Ансамбль OSNet + OSNet-AIN person-reid (torchreid): усреднённый 512-d."""

    def __init__(self, device):
        import torchreid

        def load(arch, path, nc):
            m = torchreid.models.build_model(arch, num_classes=nc, pretrained=False)
            sd = torch.load(path, map_location=device, weights_only=False)
            if 'state_dict' in sd:
                sd = sd['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            m.load_state_dict(sd)
            return m.eval().to(device)

        self.models = [
            load('osnet_x1_0', 'models/osnet_x1_0_combineall.pth', 4101),
            load('osnet_ain_x1_0', 'models/osnet_ain_x1_0_msmt17.pth', 4101),
        ]
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @torch.no_grad()
    def embed(self, crop_bgr):
        h, w = crop_bgr.shape[:2]
        if h < 40 or w < 20:
            return None
        img = cv2.resize(crop_bgr, (128, 256))
        x = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        vs = [torch.nn.functional.normalize(m(x.unsqueeze(0))[0], dim=0) for m in self.models]
        return torch.nn.functional.normalize(sum(vs), dim=0).cpu().numpy()


class Bank:
    def __init__(self, size=BANK_SIZE):
        self.items = []
        self.size = size

    def add(self, value, quality):
        if value is None or quality < MIN_QUALITY:
            return
        if len(self.items) < self.size:
            self.items.append((value, quality))
        else:
            i = int(np.argmin([q for _, q in self.items]))
            if quality > self.items[i][1]:
                self.items[i] = (value, quality)

    def mean(self):
        if not self.items:
            return None
        qs = np.array([q for _, q in self.items])
        vs = np.stack([v for v, _ in self.items])
        m = (vs * qs[:, None]).sum(axis=0) / qs.sum()
        n = np.linalg.norm(m)
        return m / n if n > 0 else None

    def total_q(self):
        return sum(q for _, q in self.items)


class Attempt:
    def __init__(self, start, tid):
        self.start = start
        self.end = start
        self.tid = tid
        self.vest = Bank()
        self.helmet = Bank()
        self.dino = Bank()
        self.boat_bins = np.zeros(18)
        self.tiles = []  # готовые тайлы для мозаики
        self.last_tile_time = -1000.0
        self.thumb_dist = None  # превью: норм. дистанция центра bbox до центра ROI
        self.thumb_t = None     # превью: время лучшего сэмпла (реальные секунды)
        self.last_embed_t = None  # время последнего OSNet-эмбеддинга (троттлинг)

    def close(self, end):
        self.end = end

    def merge(self, other):
        """Присоединить другую попытку (более позднюю) к этой."""
        self.end = other.end
        self.vest.items += other.vest.items
        self.helmet.items += other.helmet.items
        self.dino.items += other.dino.items
        self.boat_bins += other.boat_bins
        self.tiles += other.tiles
        if other.thumb_dist is not None and \
                (self.thumb_dist is None or other.thumb_dist < self.thumb_dist):
            self.thumb_dist = other.thumb_dist
            self.thumb_t = other.thumb_t


def _probe_keyframe_grid(path, fps_container, total_frames, probe_seconds=8.0):
    """Времена ключевых кадров (файловые секунды). GOP камер периодичен -
    probe первых секунд, дальше сетка арифметически. None -> fallback
    на последовательное чтение."""
    import shutil
    import subprocess
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        try:
            from utils import get_ffprobe_path
            ffprobe = get_ffprobe_path()
        except Exception:
            return None
    try:
        res = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-read_intervals', f'%+{probe_seconds}',
             '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=120)
        kts = []
        for line in res.stdout.splitlines():
            parts = line.strip().rstrip(',').split(',')
            if len(parts) == 2 and 'K' in parts[1]:
                try:
                    kts.append(float(parts[0]))
                except ValueError:
                    pass
        if len(kts) < 3:
            return None
        t0 = kts[0]
        diffs = sorted(b - a for a, b in zip(kts, kts[1:]))
        period = diffs[len(diffs) // 2]
        if period <= 0:
            return None
        duration = total_frames / fps_container if fps_container > 0 else 0.0
        if duration <= 0:
            return None
        n = int((duration - t0) / period) + 1
        return [t0 + i * period for i in range(n)]
    except Exception:
        return None


def process_video(input_path, output, roi=None, scale=0.4, no_embed=False,
                  trace=False, progress_cb=None, should_continue=None,
                  stats_out=None, slowmo_factor=1, attempt_found_cb=None,
                  log_cb=None, min_pause=None, min_attempt_duration=None,
                  collect_tiles=True, reid_embed_rate=None):
    """Обрабатывает видео: сплиттер попыток + профили атлетов.

    roi: (left, top, right, bottom) в процентах кадра.
    slowmo_factor: замедление записи (4 для слоумо 96к/с в контейнере 24к/с);
    все времена попыток возвращаются в РЕАЛЬНЫХ секундах (как в splitter.py).
    min_pause: пауза между попытками в секундах (из настроек приложения);
    определяет и закрытие попытки, и окно склейки оверсплитов.
    min_attempt_duration: минимальная длительность серии присутствия для
    подтверждения попытки (из настроек приложения).
    reid_embed_rate: сколько OSNet-эмбеддингов в секунду реального времени
    копить в профиль попытки (None/5 = каждый сэмпл; меньше = быстрее).
    progress_cb(frame_index, total_frames) — опционально.
    should_continue() -> bool — возврат False останавливает обработку.
    attempt_found_cb(attempt_dict) — вызывается, как только попытка закрыта.
    Возвращает список попыток: [{'start', 'end', 'vest', 'helmet', 'reid',
    'boat', 'boat_top', 'tiles'}].
    """
    os.makedirs(output, exist_ok=True)
    if roi is None:
        roi = ROI_PERCENTAGES

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO('models/yolo11s-seg.pt')
    embedder = None if no_embed else Embedder(device)
    if min_pause is None:
        min_pause = MIN_PAUSE_DURATION
    min_dur = MIN_ATTEMPT_DURATION if min_attempt_duration is None \
        else float(min_attempt_duration)
    merge_gap = min_pause  # пауза >= min_pause = новая попытка; меньше = оверсплит

    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened()
    fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps_container * slowmo_factor if slowmo_factor > 1 else fps_container  # реальный fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    L, T, R, B = [int(p / 100 * d) for p, d in zip(roi, (width, height, width, height))]
    roi_area = (R - L) * (B - T)

    # Сэмплирование по ключевым кадрам: декодируется только I-кадр на сэмпл
    # (вместо последовательного декодирования всех кадров H264).
    real_duration = total_frames / fps if fps > 0 else 0.0
    keyframe_times = _probe_keyframe_grid(input_path, fps_container, total_frames)
    if keyframe_times:
        kf_rate_real = len(keyframe_times) / real_duration if real_duration > 0 else 0.0
        kf_every = max(1, round(kf_rate_real / ANALYSIS_FPS)) if kf_rate_real > 0 else 1
        sample_rate = kf_rate_real / kf_every  # фактических сэмплов в реальную секунду
        sample_step = max(1, round(fps_container / ANALYSIS_FPS))  # fallback-шаг
    else:
        kf_every = 0
        sample_rate = ANALYSIS_FPS
        sample_step = max(1, round(fps_container / ANALYSIS_FPS))  # шаг в файлах-кадрах

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=int(sample_rate * 1.0),
        minimum_matching_threshold=0.8,
        frame_rate=int(sample_rate),
    )

    attempts = []
    attempt = None
    main_tid = None
    candidate_start = 0.0    # прибытие атлета: первое присутствие после длинной паузы
    candidate_confirmed = False  # создана ли попытка от текущего прибытия
    last_present_time = 0.0
    last_real_present = -1e9  # последняя РЕАЛЬНАЯ детекция атлета (не fallback)
    last_main_center = None   # центр bbox атлета при последней детекции
    absence_start = None     # время начала текущей серии отсутствия детекции
    stats = {'ticks': 0, 'main_present': 0}
    pending = []          # закрытые попытки в окне возможной склейки
    finalized = []        # сброшенные (готовые) попытки
    last_closed_end = None

    def dominant_bin(a, min_frac=0.35):
        if a.boat_bins.sum() <= 0:
            return None
        bp = a.boat_bins / a.boat_bins.sum()
        top = int(np.argmax(bp))
        return top if bp[top] >= min_frac else None

    def mergeable(a, b):
        gap = b.start - a.end
        b1, b2 = dominant_bin(a), dominant_bin(b)
        if gap > merge_gap or b1 is None or b2 is None:
            return False
        return min(abs(b1 - b2), 18 - abs(b1 - b2)) <= 1

    def make_result(a):
        vest, helmet, reid = a.vest.mean(), a.helmet.mean(), a.dino.mean()
        boat = a.boat_bins / a.boat_bins.sum() if a.boat_bins.sum() > 0 else None
        top = int(np.argmax(boat)) if boat is not None else None
        return {'start': a.start, 'end': a.end,
                'thumb_t': a.thumb_t if a.thumb_t is not None else a.start,
                'vest': vest, 'helmet': helmet, 'reid': reid,
                'boat': boat, 'boat_top': top, 'tiles': list(a.tiles)}

    def flush(a):
        nonlocal last_closed_end
        last_closed_end = max(last_closed_end or 0.0, a.end)
        if a.end - a.start < min_dur:
            msg = f'  [drop] {a.start:.1f}-{a.end:.1f}s: короче {min_dur}с'
            print(msg)
            if log_cb is not None:
                log_cb(msg)
            return
        finalized.append(a)
        if attempt_found_cb is not None:
            attempt_found_cb(make_result(a))
    def _samples():
        """Итератор сэмплов (frame, time_s, index, total).
        Если есть сетка ключевых кадров - seek к I-кадру + декод 1 кадра;
        иначе последовательное чтение с шагом sample_step."""
        if keyframe_times and kf_every > 0:
            grid = keyframe_times[::kf_every]
            total = len(grid)
            for i, kf_t in enumerate(grid):
                if should_continue is not None and not should_continue():
                    return
                if progress_cb is not None and i % 50 == 0:
                    progress_cb(i, total)
                cap.set(cv2.CAP_PROP_POS_MSEC, kf_t * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    continue
                t_real = kf_t / slowmo_factor if slowmo_factor > 1 else kf_t
                yield frame, t_real, i, total
        else:
            total = total_frames // sample_step + 1
            fi = 0
            while True:
                if should_continue is not None and not should_continue():
                    return
                ret, frame = cap.read()
                if not ret:
                    return
                if fi % sample_step == 0:
                    si = fi // sample_step
                    if progress_cb is not None and si % 50 == 0:
                        progress_cb(si, total)
                    yield frame, fi / fps, si, total
                fi += 1

    for frame, time_s, sample_i, sample_total in _samples():
        roi = frame[T:B, L:R]
        small = cv2.resize(roi, None, fx=scale, fy=scale)
        results = model(small, verbose=False)

        persons, boats = [], []  # persons: (tid, bbox, mask_bbox, rel)
        boat_dets = []
        for r in results:
            for idx, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue
                name = model.names[cls]
                x1, y1, x2, y2 = [c / scale for c in box.xyxy[0].cpu().numpy()]
                fx1, fy1, fx2, fy2 = x1 + L, y1 + T, x2 + L, y2 + T
                iw = max(min(fx2, R) - max(fx1, L), 0.0)
                ih = max(min(fy2, B) - max(fy1, T), 0.0)
                if iw <= 0 or ih <= 0:
                    continue
                if name == 'person':
                    area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
                    rel = (iw * ih) / area
                    mask = r.masks.data[idx].cpu().numpy() if r.masks is not None else None
                    persons.append({'bbox': [max(fx1, L), max(fy1, T), min(fx2, R), min(fy2, B)],
                                    'rel': rel, 'mask': mask, 'tid': None})
                elif name in BOAT_CLASSES:
                    boat_dets.append([max(fx1, L), max(fy1, T), min(fx2, R), min(fy2, B)])

        # трекинг персон (bbox уже обрезан ROI, это ок для трекера)
        if persons:
            dets = tracker.update_with_detections(
                sv.Detections(
                    xyxy=np.array([p['bbox'] for p in persons], dtype=float),
                    confidence=np.ones(len(persons)),
                    class_id=np.zeros(len(persons), dtype=int)))
            # ByteTrack может вернуть меньше детекций — сопоставляем по IoU
            for i in range(len(dets)):
                bb = dets.xyxy[i]
                best_k, best_iou = None, 0.0
                for k, p in enumerate(persons):
                    pb = p['bbox']
                    ix1, iy1 = max(bb[0], pb[0]), max(bb[1], pb[1])
                    ix2, iy2 = min(bb[2], pb[2]), min(bb[3], pb[3])
                    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
                    a1 = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    a2 = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    iou = inter / max(a1 + a2 - inter, 1e-6)
                    if iou > best_iou:
                        best_k, best_iou = k, iou
                if best_k is not None and best_iou > 0.3:
                    persons[best_k]['tid'] = int(dets.tracker_id[i])
                    persons[best_k]['bbox'] = bb.astype(int).tolist()

        # выбор главного атлета
        main = None
        if persons:
            if main_tid is not None:
                for p in persons:
                    if p['tid'] == main_tid:
                        main = p
                        break
            if main is None:
                main = max(persons, key=lambda p: p['rel'])
                main_tid = main['tid']

        if trace:
            rels = ' '.join(f"t{p['tid']}:{p['rel']:.2f}" for p in persons)
            mc = f" main=({(main['bbox'][0]+main['bbox'][2])/2:.0f},{(main['bbox'][1]+main['bbox'][3])/2:.0f})" if main else ''
            print(f'TRACE {time_s:7.2f}s main_tid={main_tid} '
                  f'attempt={attempt is not None}{"+FB" if boat_present else ""} '
                  f'persons=[{rels}]{mc}')

        main_present = main is not None and main['rel'] >= MAIN_MIN_REL
        boat_present = False
        if main_present:
            last_real_present = time_s
            mb_ = main['bbox']
            last_main_center = ((mb_[0] + mb_[2]) / 2.0, (mb_[1] + mb_[3]) / 2.0)
        elif attempt is not None and last_main_center is not None \
                and time_s - last_real_present < FALLBACK_PRESENCE_CAP and boat_dets:
            # атлет не детектируется (под водой после трюка / в брызгах),
            # но лодка видна рядом с его последней позицией - присутствие
            rw_, rh_ = max(R - L, 1), max(B - T, 1)
            best_d = None
            for bx1, by1, bx2, by2 in boat_dets:
                d = max(abs((bx1 + bx2) / 2 - last_main_center[0]) / rw_,
                        abs((by1 + by2) / 2 - last_main_center[1]) / rh_)
                if d <= 0.25 and (best_d is None or d < best_d):
                    best_d = d
            boat_present = best_d is not None
        if main_present:
            last_present_time = time_s
            # прибытие: первое присутствие после паузы >= min_pause.
            # Короткие провалы детекции (< min_pause) серию не рвут - заход
            # на плюс остаётся началом попытки даже при слабых детекциях YOLO.
            if absence_start is not None:
                if time_s - absence_start >= min_pause:
                    # сброс прибытия - только если серия уже подтвердила
                    # попытку или пауза явно длинная. Неподтверждённая серия
                    # + умеренная пауза = атлет не уходил (заход короче
                    # мин. длительности) - держим первоначальное прибытие.
                    if candidate_confirmed or time_s - absence_start > 2.0 * min_pause:
                        candidate_start = time_s
                        candidate_confirmed = False
                absence_start = None
        elif boat_present:
            if main is not None and main['rel'] < MAIN_MIN_REL:
                main = None  # залезающий краем - профиль не копим
            last_present_time = time_s
            absence_start = None
        else:
            if main is not None and main['rel'] < MAIN_MIN_REL:
                main = None  # залезающий краем — не главный
            if absence_start is None:
                absence_start = time_s
        stats['ticks'] += 1
        if main_present:
            stats['main_present'] += 1

        # машина состояний на непрерывном отсутствии: любые короткие провалы
        # детекции (< min_pause) не рвут попытку - устойчиво к редким сэмплам
        if attempt is not None:
            if main_present or boat_present:
                attempt.end = last_present_time
                if main_present:
                    attempt.tid = main_tid
            elif time_s - last_present_time >= min_pause:
                # атлет не детектируется дольше паузы - попытка закончилась
                attempt.close(last_present_time)
                pending.append(attempt)
                last_closed_end = max(last_closed_end or 0.0, attempt.end)
                attempt = None
                main_tid = None
        else:
            if main_present and time_s - candidate_start >= min_dur:
                # начало попытки = прибытие атлета (первое присутствие после
                # длинной паузы); подтверждено min_dur присутствия.
                # Если прибытие слишком давно (> 15с) - берём последние min_dur
                start_t = candidate_start if time_s - candidate_start < 15 \
                    else time_s - min_dur
                start_t = max(start_t, 0.0)
                if last_closed_end is not None and start_t < last_closed_end + 0.5:
                    start_t = last_closed_end + 0.5  # без пересечений попыток
                attempt = Attempt(start_t, main_tid)
                attempts.append(attempt)
                candidate_confirmed = True
                attempt.end = last_present_time

        # прогрессивный сброс: старая попытка не может склеиться дальше
        while pending:
            if len(pending) >= 2:
                if mergeable(pending[0], pending[1]):
                    break  # решение о склейке - в конце файла
                a = pending.pop(0)
                b1, b2 = dominant_bin(a), dominant_bin(pending[0])
                msg = (f'  [no merge] {a.start:.1f}-{a.end:.1f}s и '
                       f'{pending[0].start:.1f}-{pending[0].end:.1f}s: '
                       f'лодка {b1 * 10 if b1 is not None else "-"}°/'
                       f'{b2 * 10 if b2 is not None else "-"}°, gap={pending[0].start - a.end:.1f}с')
                print(msg)
                if log_cb is not None:
                    log_cb(msg)
                flush(a)
                continue
            if attempt is not None:
                # открытая попытка - кандидат на склейку, если началась в пределах окна
                if attempt.start - pending[0].end > merge_gap:
                    flush(pending.pop(0))
            elif main_present:
                pass  # атлет вернулся в ROI - ждём решения о склейке
            elif time_s - pending[0].end > merge_gap:
                flush(pending.pop(0))
            break

        # накопление профиля главного атлета
        if attempt is not None and main is not None:
            x1, y1, x2, y2 = [int(v) for v in main['bbox']]
            rel, mask, tid = main['rel'], main['mask'], main['tid']
            bw, bh = x2 - x1, y2 - y1
            mask_bbox = None
            if mask is not None:
                bx1s = max(int(round((x1 - L) * scale)), 0)
                by1s = max(int(round((y1 - T) * scale)), 0)
                bx2s = min(int(round((x2 - L) * scale)), mask.shape[1])
                by2s = min(int(round((y2 - T) * scale)), mask.shape[0])
                sub = mask[by1s:by2s, bx1s:bx2s]
                if sub.size:
                    mask_bbox = cv2.resize(sub.astype(np.float32), (bw, bh))

            def region(ry0, ry1, rx0, rx1):
                cy1, cy2 = int(y1 + bh * ry0), int(y1 + bh * ry1)
                cx1, cx2 = int(x1 + bw * rx0), int(x1 + bw * rx1)
                crop = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
                if crop.size == 0:
                    return None, None, 0.0
                ch, cw = crop.shape[:2]
                m = np.ones((ch, cw), np.float32)
                if mask_bbox is not None:
                    sub = mask_bbox[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
                    if sub.shape[:2] == (ch, cw):
                        m = sub
                cov = float((m > 0.5).mean())
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                sel = m > 0.5
                sat = float(hsv[..., 1][sel].mean()) / 255.0 if sel.any() else 0.0
                return crop, m, cov * (0.5 + sat)

            vc, vm, vq = region(0.25, 0.75, 0.15, 0.85)
            attempt.vest.add(hsv_hist(vc, vm), vq)
            hc, hm, hq = region(0.0, 0.22, 0.2, 0.8)
            attempt.helmet.add(hsv_hist(hc, hm), hq)

            # превью: сэмпл, где атлет ближе всего к центру ROI
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            rw, rh = max(R - L, 1), max(B - T, 1)
            d_thumb = (((bcx - (L + R) / 2) / rw) ** 2
                       + ((bcy - (T + B) / 2) / rh) ** 2) ** 0.5
            if attempt.thumb_dist is None or d_thumb < attempt.thumb_dist:
                attempt.thumb_dist = d_thumb
                attempt.thumb_t = time_s

            if embedder is not None:
                # троттлинг OSNet: не чаще reid_embed_rate раз в секунду реала
                interval = 1.0 / max(0.01, reid_embed_rate if reid_embed_rate else 5.0)
                if attempt.last_embed_t is None or \
                        time_s - attempt.last_embed_t >= interval - 1e-6:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size:
                        attempt.dino.add(embedder.embed(crop), vq)
                        attempt.last_embed_t = time_s

            # лодка: ближайшая чистая детекция к главному атлету
            best_b = None
            pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
            for bx1, by1, bx2, by2 in boat_dets:
                hb = hue_bins(frame[int(by1):int(by2), int(bx1):int(bx2)])
                if hb is None:
                    continue
                bins, sat = hb
                top = int(np.argmax(bins))
                if bins[top] < BOAT_MIN_FRAC:
                    continue
                d = np.hypot(pcx - (bx1 + bx2) / 2, pcy - (by1 + by2) / 2)
                q = np.sqrt((bx2 - bx1) * (by2 - by1)) * (0.3 + sat) / (1 + d / 200)
                if best_b is None or q > best_b[0]:
                    best_b = (q, bins)
            if best_b is not None:
                attempt.boat_bins += best_b[1] * best_b[0]

            if collect_tiles and len(attempt.tiles) < 400 and \
                    (len(attempt.tiles) == 0 or time_s - attempt.last_tile_time >= 0.4):
                c = frame[max(y1, 0):y2, max(x1, 0):x2]
                if c.size:
                    attempt.tiles.append(cv2.resize(c, (160, 120)))
                    attempt.last_tile_time = time_s

    cap.release()
    if attempt is not None:
        attempt.close(last_present_time)
        pending.append(attempt)
        last_closed_end = max(last_closed_end or 0.0, attempt.end)

    # финальная склейка соседних и сброс остатка
    while len(pending) >= 2 and mergeable(pending[0], pending[1]):
        a, b = pending.pop(0), pending.pop(0)
        msg = f'  [merge] {a.start:.1f}-{a.end:.1f}s + {b.start:.1f}-{b.end:.1f}s: одна попытка'
        print(msg)
        if log_cb is not None:
            log_cb(msg)
        a.merge(b)
        pending.insert(0, a)
    while pending:
        flush(pending.pop(0))

    # --- отчет ---
    attempts = [a for a in finalized if a.end - a.start >= min_dur]

    result = []
    for i, a in enumerate(attempts):
        r = make_result(a)
        avail = []
        if r['vest'] is not None: avail.append(f'vest(q={a.vest.total_q():.0f})')
        if r['helmet'] is not None: avail.append(f'helmet(q={a.helmet.total_q():.0f})')
        if r['reid'] is not None: avail.append('reid')
        if r['boat'] is not None:
            avail.append(f'boat({r["boat_top"] * 10}deg:{r["boat"][r["boat_top"]] * 100:.0f}%)')
        print(f'  attempt {i}: {a.start:.1f}-{a.end:.1f}s ({a.end - a.start:.1f}s) '
              f'attrs: {", ".join(avail) if avail else "none"}')
        result.append(r)

    # мозаики главных атлетов
    mosaics_dir = os.path.join(output, 'mosaics')
    if any(a['tiles'] for a in result):
        os.makedirs(mosaics_dir, exist_ok=True)
        for i, a in enumerate(result):
            if not a['tiles']:
                continue
            step = max(1, len(a['tiles']) // 6)
            tiles = a['tiles'][::step][:6]
            while len(tiles) < 6:
                tiles.append(np.zeros((120, 160, 3), np.uint8))
            mos = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
            cv2.putText(mos, f'attempt{i}_{a["start"]:.0f}-{a["end"]:.0f}s', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(mosaics_dir, f'attempt{i:02d}.jpg'), mos)

    with open(os.path.join(output, 'attempts.json'), 'w') as f:
        json.dump([{'start': r['start'], 'end': r['end']} for r in result], f, indent=1)

    # профили попыток (vest/helmet с reid в хвосте)
    pv, ph, pb, meta = [], [], [], []
    for r in result:
        v = r['vest'] if r['vest'] is not None else np.zeros(VEST_DIM)
        h = r['helmet'] if r['helmet'] is not None else np.zeros(VEST_DIM)
        d = r['reid'] if r['reid'] is not None else np.zeros(REID_DIM)
        pv.append(np.concatenate([v, d]))
        ph.append(np.concatenate([h, d]))
        pb.append(r['boat'] if r['boat'] is not None else np.zeros(18))
        meta.append([r['start'], r['end'],
                     1.0 if r['vest'] is not None else 0.0,
                     1.0 if r['helmet'] is not None else 0.0,
                     1.0 if r['reid'] is not None else 0.0,
                     1.0 if r['boat'] is not None else 0.0])
    np.savez(os.path.join(output, 'profiles.npz'),
             vest=np.array(pv), helmet=np.array(ph), boat=np.array(pb),
             meta=np.array(meta))
    print(f'\nSaved: {output}/attempts.json, profiles.npz, mosaics/')
    if stats_out is not None:
        stats_out['presence_ratio'] = stats['main_present'] / max(stats['ticks'], 1)
        stats_out['analyzed_frames'] = stats['ticks']
        stats_out['fps'] = fps
    return result


def profile_from_clip(clip_path, model=None, embedder=None, scale=0.4,
                      max_samples=12, max_crops=3):
    """Профиль попытки из готового клипа (короткая схема): до max_crops лучших
    кропов атлета (YOLO-seg + OSNet) по всему клипу -> vest HSV + reid + лодка.
    Для кросс-папочной детекции без пересканирования исходников.
    Возвращает {'vest': 1664d, 'boat': 18d} или None, если атлета нет."""
    if model is None:
        model = YOLO('models/yolo11s-seg.pt')
    if embedder is None:
        embedder = Embedder(default_device())
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        return None
    step = max(1, round(n_frames / max_samples))
    crops = []          # (score, crop, mask) - кандидаты кропа атлета
    boat_best = None    # (q, bins18)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step == 0:
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            H, W = frame.shape[:2]
            for r in model(small, verbose=False):
                for idx, box in enumerate(r.boxes):
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    if conf < YOLO_CONFIDENCE_THRESHOLD:
                        continue
                    cls_name = model.names[cls]
                    x1, y1, x2, y2 = [c / scale for c in box.xyxy[0].cpu().numpy()]
                    x1, y1 = max(int(x1), 0), max(int(y1), 0)
                    x2, y2 = min(int(x2), W), min(int(y2), H)
                    if x2 - x1 < 30 or y2 - y1 < 60:
                        continue
                    if cls_name == 'person':
                        mb = None
                        if r.masks is not None:
                            m = r.masks.data[idx].cpu().numpy()
                            sub = m[max(int(y1 * scale), 0):int(y2 * scale),
                                    max(int(x1 * scale), 0):int(x2 * scale)]
                            if sub.size:
                                mb = cv2.resize(sub.astype(np.float32),
                                                (x2 - x1, y2 - y1))
                        area = (x2 - x1) * (y2 - y1)
                        crops.append((conf * np.sqrt(area),
                                      frame[y1:y2, x1:x2], mb))
                    elif cls_name in BOAT_CLASSES:
                        hb = hue_bins(frame[y1:y2, x1:x2])
                        if hb is None:
                            continue
                        bins, sat = hb
                        if bins[int(np.argmax(bins))] < BOAT_MIN_FRAC:
                            continue
                        q = conf * (0.3 + sat)
                        if boat_best is None or q > boat_best[0]:
                            boat_best = (q, bins)
        fi += 1
    cap.release()
    crops.sort(key=lambda t: -t[0])
    crops = crops[:max_crops]
    if not crops:
        return None
    reid = [v for v in (embedder.embed(c) for _, c, _ in crops) if v is not None]
    if not reid:
        return None
    rmean = np.mean(reid, axis=0)
    rmean = rmean / max(np.linalg.norm(rmean), 1e-9)
    hists = [h for h in (hsv_hist(c, m) for _, c, m in crops) if h is not None]
    seg = np.zeros(VEST_DIM)
    if hists:
        vm = np.mean(hists, axis=0)
        nv = np.linalg.norm(vm)
        if nv > 0:
            seg = vm / nv
    full = np.concatenate([seg, rmean])
    full = full / max(np.linalg.norm(full), 1e-9)
    boat = boat_best[1] / boat_best[1].sum() if boat_best is not None else np.zeros(18)
    return {'vest': full, 'boat': boat}


def main():
    args = parse_arguments()
    process_video(args.input, args.output, roi=args.roi, scale=args.scale,
                  no_embed=args.no_embed, trace=args.trace)


if __name__ == '__main__':
    main()
