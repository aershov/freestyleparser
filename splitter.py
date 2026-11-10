import cv2
import i18n
import numpy as np
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from ultralytics import YOLO

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Порог уверенности для детекции объектов
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 600  # уменьшаем кадр перед анализом в YOLO до стольких пикселей

# Конфигурация обработки видео
TARGET_ANALYSIS_FPS = 5  # Целевая частота анализа (сэмплов в секунду реального времени)
MIN_DETECTION_TIME = 0.6  # Минимальное время (в секундах) с детекциями для начала попытки

# Режим сканирования (ручной выбор попыток): значения по умолчанию,
# настраиваются в UI через params 'scan_interval' и 'scan_threshold'
SCAN_INTERVAL = 2.5     # Шаг сканирования, секунд реального времени между сэмплами
SCAN_CONFIDENCE = 0.25  # Пониженный порог YOLO для сканирования

# Скан по ключевым кадрам: если между I-кадрами в среднем больше этой паузы,
# ключевые кадры слишком редки для детекции - откат к обычным сэмплам
KEYFRAME_MAX_GAP = 5.0
# Скан всегда по ключевым кадрам (I-кадрам); переключатель убран из настроек
KEYFRAME_SCAN = True
# Стриминговый скан: столько интервалов между I-кадрами потребляется на
# бутстрапе для оценки GOP (кадры буферизуются и отдаются далее)
KEYFRAME_BOOTSTRAP_GAPS = 8

np.seterr(divide='ignore', invalid='ignore')

# переэкспорт: живёт в utils.py, чтобы app.py мог импортировать дефолты
# без тяжёлых torch/ultralytics при старте
from utils import DEFAULT_PROCESSING_PARAMS, get_ffmpeg_path  # noqa: E402

_LOG_FN = None


def _log(msg):
    """Лог скана: в GUI-лог приложения (log_fn), иначе в консоль."""
    if _LOG_FN is not None:
        _LOG_FN(msg)
    else:
        safe_print(msg)

class AttemptInfo:

    def __init__(self, source_video, start, end, number, best_frame, person_frame, base_frame, person_bbox,
                 is_candidate=False, thumb_t=None, thumb_frame=None):
        self.source_video = source_video
        self.start = start
        self.end = end
        self.number = number
        self.best_frame = best_frame
        self.person_frame = person_frame
        self.base_frame = base_frame
        self.person_bbox = person_bbox
        self.is_candidate = is_candidate
        # превью: момент и кадр, где атлет ближе всего к центру ROI
        self.thumb_t = thumb_t
        self.thumb_frame = thumb_frame

    def duration(self):
        return self.end - self.start


def safe_print(*args, **kwargs):
    """Потокобезопасный вывод в консоль"""
    print(*args, **kwargs)


def _iou(box1, box2):
    """IoU между двумя bbox [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / (a1 + a2 - inter)


_YOLO_MODEL = None

# Кэш проб скорости (декодинг файла / модель на эталоне), за сессию
_PROBE_CACHE = {}


def load_yolo_model():
    """Загружает модель YOLO с кешем: загрузка занимает секунды,
    переиспользуем между файлами/потоками (модель stateless)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/yolo11s-seg.pt')
        _YOLO_MODEL = YOLO(model_path)
        # явный выбор ускорителя: без этого ultralytics оставляет модель
        # на cpu даже на Apple Silicon (MPS ~в 3.6 раза быстрее на кадр)
        try:
            import torch
            dev = 'cuda' if torch.cuda.is_available() else \
                ('mps' if torch.backends.mps.is_available() else 'cpu')
            if dev != 'cpu':
                _YOLO_MODEL.to(dev)
                # прогрев: MPS-компиляция графика на первом инференсе (~10с),
                # без неё первый сэмпл анализа тормозит весь прогресс
                _YOLO_MODEL(np.zeros((338, 600, 3), dtype=np.uint8), verbose=False)
        except Exception:
            pass
        # устройство инференса в лог: cpu на машинах без GPU = медленно,
        # пользователь должен видеть причину из лога
        try:
            dev = next(_YOLO_MODEL.model.parameters()).device
            _log(i18n.tr("Нейросеть YOLO загружена (yolo11s-seg), устройство: {dev}",
                         dev=dev))
        except Exception:
            pass
    return _YOLO_MODEL


def _finalize_attempt(attempt_number, start_time, end_time, real_duration,
                      person_bbox, base_frame, best_frame, person_frame,
                      fallback_frame, video_file, callback, params, scan_mode=False,
                      thumb_t=None, thumb_frame=None):
    """Создаёт попытку и вызывает callback. Возвращает (attempt, attempt_number) или (None, attempt_number) или (False, attempt_number) для стопа.
    scan_mode: границы отдаются «сырыми» (без паддинга) - его добавляет
    сохраняющий код (ручная нарезка / автосохранение) единообразно."""
    if scan_mode:
        # единая конвенция скан-режимов: детектор отдаёт сырые границы,
        # паддинг из params добавляет код сохранения один раз
        params = dict(params, attempt_start_padding=0.0, attempt_end_padding=0.0)
    start_padding = params['attempt_start_padding']
    end_padding = params['attempt_end_padding']
    min_duration = params['min_attempt_duration']

    start1 = max(0, start_time - start_padding)
    end1 = min(end_time + end_padding, real_duration)

    _skip_word = i18n.tr("Зона") if scan_mode else i18n.tr("Попытка")
    if end1 <= start1:
        _log(i18n.tr("  [SKIP] {word} #{n}: start ({start:.2f}s) >= end ({end:.2f}s)",
                           word=_skip_word, n=attempt_number, start=start1, end=end1))
        return None, attempt_number

    duration = end1 - start1
    if duration < min_duration:
        _log(i18n.tr("  [SKIP] {word} #{n}: duration {dur:.2f}s < {min}s",
                           word=_skip_word, n=attempt_number, dur=duration, min=min_duration))
        return None, attempt_number

    if person_bbox is None:
        _log(i18n.tr("  [SKIP] {word} #{n}: нет данных о позиции человека", word=_skip_word, n=attempt_number))
        return None, attempt_number

    x1, y1, x2, y2 = map(int, person_bbox)
    try:
        bf = base_frame[y1:y2, x1:x2]
    except:
        bf = fallback_frame[y1:y2, x1:x2]

    attempt = AttemptInfo(number=attempt_number,
                          base_frame=bf,
                          best_frame=best_frame,
                          person_frame=person_frame,
                          person_bbox=person_bbox,
                          source_video=video_file,
                          start=start1, end=end1,
                          is_candidate=scan_mode,
                          thumb_t=thumb_t,
                          thumb_frame=thumb_frame)
    attempt_number += 1
    need_stop = not callback(attempt)
    if need_stop:
        return False, attempt_number
    return attempt, attempt_number


def open_capture(video_file):
    """Открывает видео. На Windows бэкенд FFMPEG часто не читает файлы камер
    (например, GoPro: чтение обрывается на первых кадрах с "packet read max
    attempts exceeded"), поэтому сначала пробуем Media Foundation.
    Если путь с не-ASCII (кириллица в имени пользователя) и открыть не
    удалось - копируем во временную папку с ASCII-именем."""
    cap = None
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(video_file, cv2.CAP_MSMF)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened():
        return cap
    if sys.platform == 'win32':
        try:
            video_file.encode('ascii')
            return cap  # путь ASCII - проблема не в нём
        except UnicodeEncodeError:
            pass
        try:
            import hashlib
            import shutil as _sh
            import tempfile as _tf
            tmp_dir = os.path.join(_tf.gettempdir(), 'FreestyleParserCaps')
            os.makedirs(tmp_dir, exist_ok=True)
            digest = hashlib.md5(video_file.encode('utf-8')).hexdigest()[:12]
            tmp_path = os.path.join(tmp_dir, digest + os.path.splitext(video_file)[1])
            if not os.path.exists(tmp_path) or \
                    os.path.getsize(tmp_path) != os.path.getsize(video_file):
                _sh.copy2(video_file, tmp_path)
            fallback = cv2.VideoCapture(tmp_path, cv2.CAP_MSMF)
            if fallback.isOpened():
                return fallback
            fallback.release()
            return cv2.VideoCapture(tmp_path)
        except Exception:
            pass
    return cap


def _seek_sample_iter(cap, fps_real, total_frames, sample_rate, range_start, range_end):
    """Генератор сэмплов точечными seek'ами. Для редкой выборки (скан) намного
    быстрее полного декодирования: каждый seek декодирует не больше кадров,
    чем GOP. sample_rate - сэмплов в секунду реального времени."""
    step = max(1, int(round(fps_real / sample_rate)))
    _log(i18n.tr("  Скан точечным seek: шаг {step} кадров, ~{rate:.1f}/с реальных",
                 step=step, rate=sample_rate))
    start_f = int(range_start * fps_real) if range_start and range_start > 0 else 0
    end_f = total_frames if range_end is None else min(total_frames, int(range_end * fps_real))
    seek_times = []
    seek_warned = False
    pos = start_f
    while pos < end_f:
        t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break
        if len(seek_times) < 3:
            seek_times.append(time.perf_counter() - t0)
            if len(seek_times) == 3 and not seek_warned:
                seek_warned = True
                avg = sum(seek_times) / len(seek_times)
                if avg > 1.5:
                    remaining = max(0, (end_f - pos) // step)
                    _log(i18n.tr("  Точечный seek медленный ({t:.1f} с/сэмпл): "
                                 "прогноз ~{h:.1f} ч на этот файл",
                                 t=avg, h=avg * remaining / 3600))
        yield pos / fps_real, frame, pos, end_f
        pos += step

def _grab_sample_iter(cap, fps_container, fps_real, total_frames, frame_skip,
                      range_start, range_end):
    """Генератор сэмплов из полного декодирования (как раньше).
    Декодируется каждый frame_skip-й кадр; сэмплы в реальных секундах."""
    _log(i18n.tr("  FPS: {container:.1f}, реальный fps: {real:.1f}, "
                       "frame_skip: {skip}, анализ: {rate:.1f}/с",
                       container=fps_container, real=fps_real, skip=frame_skip,
                       rate=fps_real / frame_skip))
    start_frame = 0
    if range_start and range_start > 0:
        start_frame = int(range_start * fps_real)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _log(i18n.tr("  Старт с кадра {frame} (диапазон от {start:.1f}s)",
                               frame=start_frame, start=range_start))
    frame_index = start_frame
    while True:
        if not cap.grab():
            break
        frame_index += 1
        if frame_index >= total_frames:
            break
        if frame_index % frame_skip:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            break
        real_sec = frame_index / fps_real
        if range_end is not None and real_sec > range_end:
            break
        yield real_sec, frame, frame_index, total_frames


_PTS_TIME_RE = re.compile(r'pts_time:([0-9]+(?:\.[0-9]+)?)')


def _median_gap(pts):
    """Медианный интервал между метками времени (с)."""
    gaps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


class _KeyframeStreamIter:
    """Стриминговый скан по ключевым кадрам: ffmpeg декодирует только I-кадры
    (-skip_frame nokey) и льёт их в stdout-пайп, потребитель (YOLO) читает
    по мере поступления. Если нейросеть медленнее декодирования - ffmpeg
    блокируется на записи в пайп (backpressure): процессинг "догоняет"
    извлечение. Промежуточных jpg нет, ожидания полного прохода нет -
    первый сэмпл готов через секунды после старта.
    Бутстрап: первые KEYFRAME_BOOTSTRAP_GAPS интервалов между I-кадрами
    потребляются внутрь (кадры буферизуются, потом отдаются) для оценки GOP;
    до этого берутся все I-кадры - перестраховка в пользу плотности. После
    бутстрапа выставляются rate/sample_count, дальше сэмплы выбираются на
    лету с порогом (шаг - 0.5*GOP) - шаг в интервале [шаг - 0.5*GOP;
    шаг + 0.5*GOP), без раздвоения при GOP чуть меньше шага.
    Если GOP > KEYFRAME_MAX_GAP, ffmpeg недоступен или пайп сорвался -
    откат к seek-скану по тому же cap."""

    def __init__(self, cap, video_file, fps_real, total_frames, real_duration,
                 scan_interval, slowmo_factor, range_start, range_end,
                 sample_rate, seek_pos=None):
        self.cap = cap
        self.video_file = video_file
        self.fps_real = fps_real
        self.total_frames = total_frames
        self.real_duration = real_duration
        self.scan_interval = scan_interval
        self.slowmo_factor = slowmo_factor
        self.range_start = range_start
        self.range_end = range_end
        self.sample_rate = sample_rate
        self.seek_pos = seek_pos        # старт декодирования с позиции (резюме)
        self.rate = sample_rate          # уточняется после бутстрапа
        self.sample_count = int(real_duration * sample_rate) + 1
        self._h_out = None
        self._frame_bytes = 0
        self._proc = None
        self._pts_q = None
        self._reader = None
        self._taken = 0

    def _start_ffmpeg(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ffmpeg = get_ffmpeg_path()
        if w <= 0 or h <= 0 or not ffmpeg:
            return False
        self._h_out = max(2, int(round(h * YOLO_MAX_WIDTH / w / 2.0)) * 2)
        self._frame_bytes = YOLO_MAX_WIDTH * self._h_out * 3
        cmd = [ffmpeg]
        if self.seek_pos:
            # Резюме: вход с позиции seek_pos; -copyts сохраняет абсолютные
            # pts контейнера (без пересчёта с нуля) - реальные секунды сходятся
            cmd += ['-ss', f'{self.seek_pos:.3f}', '-copyts']
        cmd += ['-hide_banner', '-nostdin', '-skip_frame', 'nokey',
                '-i', self.video_file,
               '-vf', f'showinfo,scale={YOLO_MAX_WIDTH}:{self._h_out}',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vsync', '0', 'pipe:1']
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        except Exception:
            self._proc = None
            return False
        self._pts_q = queue.Queue()

        def _reader():
            self._stderr_tail = []
            try:
                try:
                    for line in self._proc.stderr:
                        text = line.decode('utf-8', 'replace')
                        # хвост stderr - для диагностики, если ffmpeg не дал кадров
                        self._stderr_tail.append(text.strip())
                        if len(self._stderr_tail) > 5:
                            self._stderr_tail.pop(0)
                        m = _PTS_TIME_RE.search(text)
                        if m:
                            try:
                                self._pts_q.put(float(m.group(1)))
                            except ValueError:
                                pass
                except Exception:
                    pass  # stderr закрыт при terminate - штатно
            finally:
                self._pts_q.put(None)

        self._reader = threading.Thread(target=_reader, daemon=True)
        self._reader.start()
        return True

    def _read_frame(self):
        """Читает ровно один кадр из пайпа; None - конец потока или сбой."""
        buf = bytearray()
        try:
            while len(buf) < self._frame_bytes:
                chunk = self._proc.stdout.read(self._frame_bytes - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            pts = self._pts_q.get(timeout=120)
        except Exception:
            return None
        if pts is None:
            return None
        frame = np.frombuffer(bytes(buf), dtype=np.uint8)
        return pts, frame.reshape(self._h_out, YOLO_MAX_WIDTH, 3)

    def _real_sec(self, pts):
        # с -copyts pts всегда абсолютные (контейнерные)
        return pts / self.slowmo_factor if self.slowmo_factor > 1 else pts

    def _in_range(self, real_sec):
        if self.range_start is not None and real_sec < self.range_start:
            return False
        if self.range_end is not None and real_sec > self.range_end:
            return False
        return True

    def _stop_ffmpeg(self):
        proc, self._proc = self._proc, None
        self._reader = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            for s in (proc.stdout, proc.stderr):
                try:
                    s.close()
                except Exception:
                    pass

    def __iter__(self):
        if not self._start_ffmpeg():
            _log(i18n.tr("  Не удалось извлечь ключевые кадры - скан обычными сэмплами"))
            yield from _seek_sample_iter(self.cap, self.fps_real, self.total_frames,
                                         self.sample_rate, self.range_start,
                                         self.range_end)
            return
        try:
            # Бутстрап: потребляем первые I-кадры, пока не накопим достаточно
            # интервалов для оценки GOP; кадры буферизуются и отдаются далее
            buffered = []
            gaps = []
            last_pts = None
            while len(gaps) < KEYFRAME_BOOTSTRAP_GAPS:
                item = self._read_frame()
                if item is None:
                    break
                pts, frame = item
                buffered.append((pts, frame))
                if last_pts is not None and pts > last_pts:
                    gaps.append(pts - last_pts)
                last_pts = pts
            gop = _median_gap([pts for pts, _ in buffered])
            if len(buffered) < 2 or gop <= 0:
                # ffmpeg не отдал ни одной пары I-кадров (сборка не умеет
                # -skip_frame nokey для этого контейнера, битый поток и т.п.) -
                # стриминг здесь не работает, откат к seek-скану
                tail = ' | '.join(getattr(self, '_stderr_tail', []))
                _log(i18n.tr("  Внимание: ключевые кадры не читаются "
                             "(получено кадров: {n}, GOP ~{gop:.2f}с) - "
                             "скан обычными сэмплами (seek)", n=len(buffered), gop=gop))
                if tail:
                    _log(i18n.tr("  ffmpeg: {tail}", tail=tail))
                self._stop_ffmpeg()
                yield from _seek_sample_iter(self.cap, self.fps_real,
                                             self.total_frames, self.sample_rate,
                                             self.range_start, self.range_end)
                return
            if gop > KEYFRAME_MAX_GAP:
                _log(i18n.tr("  Ключевые кадры редкие (шаг ~{gap:.1f}с > {max:.0f}с) - "
                             "скан обычными сэмплами", gap=gop, max=KEYFRAME_MAX_GAP))
                yield from _seek_sample_iter(self.cap, self.fps_real,
                                             self.total_frames, self.sample_rate,
                                             self.range_start, self.range_end)
                return
            threshold = max(0.0, self.scan_interval - gop * 0.5)
            # Оценка эффективного шага: жадный отбор по сетке I-кадров даёт
            # шаг в [threshold; threshold+GOP); матожидание - threshold+GOP/2
            eff = max(gop, threshold + gop * 0.5)
            if eff > 0:
                self.rate = 1.0 / eff
            if self.real_duration > 0 and eff > 0:
                self.sample_count = max(2, int(self.real_duration / eff) + 1)
            _log(i18n.tr("  Скан по ключевым кадрам (стриминг): шаг ~{step:.2f}с, "
                         "~{n} сэмплов (I-кадры каждые {gop:.2f}с)",
                         step=eff, n=self.sample_count, gop=gop))
            if gop > self.scan_interval + 1e-6:
                _log(i18n.tr("  Внимание: I-кадры ({gop:.2f}с) реже шага скана "
                             "({si:.1f}с) - детекция грубее, возможны пропуски",
                             gop=gop, si=self.scan_interval))
                _log(i18n.tr("  Рекомендация: шаг скана {step} для этого файла "
                             "(I-кадры каждые {gop:.2f}с)",
                             step=_recommended_scan_step(gop), gop=gop))
            last_t = None
            for pts, frame in buffered:
                if last_t is None or pts - last_t >= threshold - 1e-6:
                    last_t = pts
                    real_sec = self._real_sec(pts)
                    if self._in_range(real_sec):
                        self._taken += 1
                        yield real_sec, frame, self._taken, self.sample_count
            del buffered
            while True:
                item = self._read_frame()
                if item is None:
                    break
                pts, frame = item
                if last_t is not None and pts - last_t < threshold - 1e-6:
                    continue
                last_t = pts
                real_sec = self._real_sec(pts)
                if not self._in_range(real_sec):
                    if self.range_end is not None and real_sec > self.range_end:
                        break
                    continue
                self._taken += 1
                yield real_sec, frame, self._taken, self.sample_count
        finally:
            self._stop_ffmpeg()


def _recommended_scan_step(gop):
    """Рекомендованный шаг скана для файла с данным GOP (из выбора UI)."""
    if gop >= 1.5:
        return '2'
    if gop >= 0.75:
        return '1'
    return '1/2'


class _Detector:
    """Машина состояний детекции попыток на потоке сэмплов (реальное время + кадр)."""

    def __init__(self, model, roi, video_file, analysis_rate, real_duration,
                 params, min_strength, attempt_number, callback, scan_mode=False,
                 range_start=None, range_end=None):
        self.model = model
        self.roi = roi
        self.video_file = video_file
        self.rate = analysis_rate  # сэмплов в секунду реального времени
        self.real_duration = real_duration
        self.params = params
        self.min_strength = min_strength
        self.attempt_number = attempt_number
        self.callback = callback
        self.scan_mode = scan_mode
        self.range_start = range_start
        self.range_end = range_end

        self.conf_threshold = (float(params.get('scan_threshold', SCAN_CONFIDENCE))
                               if scan_mode else YOLO_CONFIDENCE_THRESHOLD)
        self.min_pause = params['min_pause_duration']
        self.window_size = max(2, int(round(MIN_DETECTION_TIME * self.rate)))
        self.pause_samples = max(1, self.min_pause * self.rate)  # float, как в оригинале

        self.in_attempt = False
        self.start_time = None
        self.end_time = None
        self.frames_in_attempt = 0
        self.frames_out_of_attempt = 0
        self.best_frame = None
        self.best_frame_confidence = 0
        self.person_frame = None
        self.base_frame = None
        self.person_bbox = None
        self.person_conf = 0
        self.last_person_bbox = None  # для fallback bbox когда person пропадает в брызгах
        self.fallback_frame = None

        self.detection_window = deque(maxlen=self.window_size)
        self.bbox_history = deque(maxlen=self.window_size)  # [(x1,y1,x2,y2,conf), ...]

        # превью: сэмпл, где атлет ближе всего к центру ROI (единая логика
        # для всех режимов); сравниваем в нормированных координатах сэмпла
        self.thumb_dist = None
        self.thumb_t = None
        self.thumb_frame = None
        self.roi_center_norm = self._roi_center_norm(roi)

    @staticmethod
    def _roi_center_norm(roi):
        """Центр ROI в нормированных координатах сэмпла (0..1).
        Прямоугольный ROI: сэмпл уже обрезан по ROI - центр (0.5, 0.5).
        Полигон: сэмпл - полный кадр, центр = среднее точек (в %)."""
        if roi and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in roi):
            xs = [float(p[0]) for p in roi]
            ys = [float(p[1]) for p in roi]
            return (sum(xs) / len(xs) / 100.0, sum(ys) / len(ys) / 100.0)
        return (0.5, 0.5)

    def process_sample(self, current_sec, frame):
        """Обрабатывает один сэмпл. Возвращает AttemptInfo (попытка завершена),
        False (остановка по запросу callback) или None."""
        original_frame = frame.copy()

        # Установка области интереса
        if self.roi:
            frame_height, frame_width = frame.shape[:2]
            # Прямоугольник - 4 скаляра; полигон из 4 точек тоже даёт len==4,
            # поэтому различаем по типу элементов
            if len(self.roi) == 4 and all(isinstance(v, (int, float)) for v in self.roi):
                left, top, right, bottom = self.roi
                x1_px = int(frame_width * left / 100)
                y1_px = int(frame_height * top / 100)
                x2_px = int(frame_width * right / 100)
                y2_px = int(frame_height * bottom / 100)
                frame = frame[y1_px:y2_px, x1_px:x2_px]
                original_frame = original_frame[y1_px:y2_px, x1_px:x2_px]
            elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in self.roi):
                # Многоугольник
                polygon_points = [(int(frame_width * x / 100), int(frame_height * y / 100)) for x, y in self.roi]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                polygon_array = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_array], 255)
                frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Масштабирование для YOLO
        if frame.shape[1] > YOLO_MAX_WIDTH:
            scale_factor = YOLO_MAX_WIDTH / frame.shape[1]
            small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            small_frame_original = cv2.resize(original_frame, None, fx=scale_factor, fy=scale_factor)
        else:
            small_frame = frame.copy()
            small_frame_original = original_frame.copy()
        self.fallback_frame = small_frame_original

        results = self.model(small_frame, verbose=False)
        self.last_results = results  # для профайлера атлетов (без 2го инференса)

        # Проверка наличия объектов в ТЕКУЩЕМ кадре
        person_detected_in_frame = False
        best_conf_in_frame = 0
        best_box_in_frame = None
        best_person_box = None
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                if class_name in YOLO_TARGET_CLASSES and conf >= self.conf_threshold:
                    person_detected_in_frame = True
                    if conf > best_conf_in_frame:
                        best_conf_in_frame = conf
                        best_box_in_frame = box
                if class_name == 'person' and conf > self.person_conf:
                    best_person_box = box
                    self.person_conf = conf

        # Обновляем лучший bbox для превью и историю
        if best_box_in_frame is not None:
            best_box_np = best_box_in_frame.xyxy[0].cpu().numpy()
            bx1, by1, bx2, by2 = map(int, best_box_np)

            # Отслеживаем person bbox отдельно (fallback для превью)
            if best_person_box is not None:
                self.last_person_bbox = best_person_box.xyxy[0].cpu().numpy()
                best_person_box_np = self.last_person_bbox
            else:
                best_person_box_np = self.last_person_bbox if self.last_person_bbox is not None else best_box_np

            self.person_bbox = best_person_box_np
            self.bbox_history.append((bx1, by1, bx2, by2, best_conf_in_frame))

            # Лучший кадр превью: атлет ближе всего к центру ROI
            fh, fw = small_frame_original.shape[:2]
            cx = (best_person_box_np[0] + best_person_box_np[2]) / (2.0 * fw)
            cy = (best_person_box_np[1] + best_person_box_np[3]) / (2.0 * fh)
            d = ((cx - self.roi_center_norm[0]) ** 2 +
                 (cy - self.roi_center_norm[1]) ** 2) ** 0.5
            if self.thumb_dist is None or d < self.thumb_dist:
                self.thumb_dist = d
                self.thumb_t = current_sec
                self.thumb_frame = small_frame_original

            # Обновляем лучший кадр для превью (приоритет к person)
            if best_person_box is not None:
                ppx1, ppy1, ppx2, ppy2 = map(int, best_person_box_np)
                self.person_frame = small_frame_original[ppy1:ppy2, ppx1:ppx2]
            else:
                self.person_frame = small_frame_original[by1:by2, bx1:bx2]

            if best_conf_in_frame > self.best_frame_confidence:
                self.best_frame_confidence = best_conf_in_frame
                self.best_frame = small_frame_original

            if self.person_conf > 0.5:
                self.best_frame_confidence = self.person_conf
                self.best_frame = small_frame_original

        elif self.best_frame is None and small_frame_original is not None:
            self.best_frame = small_frame_original
            self.person_frame = small_frame_original

        # Вычисляем detection strength: bbox motion (IoU) + confidence drift
        detection_strength = 0.0
        if person_detected_in_frame and len(self.bbox_history) >= 2:
            ious = []
            confs = []
            for i in range(1, len(self.bbox_history)):
                prev = self.bbox_history[i - 1]
                curr = self.bbox_history[i]
                ious.append(_iou(prev[0:4], curr[0:4]))
                confs.append(abs(curr[4] - prev[4]))
            avg_iou = sum(ious) / len(ious)
            avg_conf_drift = sum(confs) / len(confs)
            detection_strength = (1.0 - avg_iou) * 0.7 + min(avg_conf_drift * 10, 1.0) * 0.3

        # Hysteresis: во время попытки порог strength ниже (ловим брызги/переходные моменты)
        actual_min_strength = self.min_strength * 0.3 if self.in_attempt else self.min_strength
        has_detections = person_detected_in_frame and detection_strength >= actual_min_strength

        # запоминаем базовый фрейм без человека
        if not self.in_attempt:
            if self.base_frame is None:
                self.base_frame = small_frame_original

        if has_detections:
            self.detection_window.append(1)
            self.frames_in_attempt += 1
            self.frames_out_of_attempt = 0
            if not self.in_attempt:
                # Debounce: требуем стабильных детекций перед стартом попытки
                if len(self.detection_window) >= self.window_size and all(self.detection_window):
                    detection_start_time = current_sec
                    if detection_start_time - (self.end_time or 0) >= self.min_pause:
                        self.in_attempt = True
                        self.start_time = detection_start_time
                        if self.scan_mode:
                            # скан-режим: считаются все зоны, включая потом
                            # отброшенные как дубли области - номер служебный
                            _log(i18n.tr("  [START] Зона #{n} на {t:.2f}s",
                                         n=self.attempt_number, t=self.start_time))
                        else:
                            _log(i18n.tr("  [START] Попытка #{n} на {t:.2f}s",
                                         n=self.attempt_number, t=self.start_time))
        else:
            self.detection_window.append(0)
            self.frames_out_of_attempt += 1
            if self.in_attempt and self.frames_out_of_attempt >= self.pause_samples:
                self.in_attempt = False
                self.best_frame_confidence = 0
                self.detection_window.clear()
                self.bbox_history.clear()
                self.person_conf = 0
                self.last_person_bbox = None
                self.end_time = current_sec
                return self._finalize(self.start_time, self.end_time)

        return None

    def _finalize(self, start_time, end_time):
        """Финализирует попытку с учётом диапазона и длительности файла."""
        end_limit = self.real_duration
        if self.range_end is not None:
            end_limit = min(end_limit, self.range_end)
        result, self.attempt_number = _finalize_attempt(
            self.attempt_number, start_time, end_time, end_limit,
            self.person_bbox, self.base_frame, self.best_frame, self.person_frame,
            self.fallback_frame, self.video_file, self.callback, self.params,
            scan_mode=self.scan_mode,
            thumb_t=self.thumb_t, thumb_frame=self.thumb_frame)
        if result is not None:
            self.best_frame = None
            self.base_frame = None
            self.person_bbox = None
            self.thumb_dist = None
            self.thumb_t = None
            self.thumb_frame = None
        return result

    def finish(self):
        """Финализирует незакрытую попытку на конце файла/диапазона."""
        if self.in_attempt and self.start_time is not None:
            end_time = self.real_duration if self.range_end is None else min(self.real_duration, self.range_end)
            if self.scan_mode:
                _log(i18n.tr("  [{file}] Финализация зоны #{n} на конце ({t:.1f}s)",
                             file=os.path.basename(self.video_file),
                             n=self.attempt_number, t=end_time))
            else:
                _log(i18n.tr("  [{file}] Финализация попытки #{n} на конце ({t:.1f}s)",
                             file=os.path.basename(self.video_file),
                             n=self.attempt_number, t=end_time))
            return self._finalize(self.start_time, end_time)
        return None

def _reference_video_path():
    """Эталонный ролик для пробы скорости скана (dev: assets/, сборка:
    распакованные ресурсы PyInstaller)."""
    import sys
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'assets', 'reference_scan.mp4')


def probe_stream_rate(video_path, budget_sec=4.0):
    """Проба декодинга I-кадров БЕЗ модели: носитель + декодер конкретного
    файла (флешка/диск, GOP, разрешение). Возвращает dict
    {'rate': video_sec/wall_sec, 'density': сэмплов на сек видео} или None.
    Кэш за сессию по файлу."""
    key = ('stream', video_path)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        cap = open_capture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        real_duration = total_frames / fps if fps > 0 else 0.0
        stream = _KeyframeStreamIter(cap, video_path, fps, total_frames,
                                     real_duration, 1.0, 1, None, None, 1.0)
        t0 = time.time()
        first_sec = last_sec = None
        samples = 0
        for real_sec, _frame, _pos, _tot in iter(stream):
            if first_sec is None:
                first_sec = real_sec
            last_sec = real_sec
            samples += 1
            if time.time() - t0 >= budget_sec:
                break
        wall = max(0.2, time.time() - t0)
        video_sec = max(0.0, last_sec - first_sec) if first_sec is not None else 0.0
        res = None
        if video_sec > 1.0:
            res = {'rate': video_sec / wall,
                   'density': samples / video_sec}
        _PROBE_CACHE[key] = res
        return res
    except Exception:
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def probe_model_rate(budget_sec=4.0):
    """Проба модели на эталонном ролике: полный пайплайн (YOLO-детекция)
    на детерминированном CPU-пути. Возвращает dict
    {'rate': video_sec/wall_sec, 'density': сэмплов на сек видео} или None.
    Кэш за сессию."""
    key = ('model',)
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    ref = _reference_video_path()
    if not os.path.exists(ref):
        return None
    model = load_yolo_model()  # загрузка в кэш; в пробу не включаем
    # прогрев: первый инференс компилирует MPS-граф (~10с) - вне замера
    try:
        model(np.zeros((338, 600, 3), dtype=np.uint8), verbose=False)
    except Exception:
        pass
    try:
        cap = open_capture(ref)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        real_duration = total_frames / fps if fps > 0 else 0.0
        stream = _KeyframeStreamIter(cap, ref, fps, total_frames, real_duration,
                                     1.0, 1, None, None, 1.0)
        detector = _Detector(load_yolo_model(), (0, 0, 100, 100), ref,
                             analysis_rate=stream.rate,
                             real_duration=real_duration,
                             params=DEFAULT_PROCESSING_PARAMS.copy(),
                             min_strength=0.5, attempt_number=1,
                             callback=lambda a: True, scan_mode=True)
        t0 = time.time()
        first_sec = last_sec = None
        samples = 0
        for real_sec, frame, _pos, _tot in iter(stream):
            detector.process_sample(real_sec, frame)
            if first_sec is None:
                first_sec = real_sec
            last_sec = real_sec
            samples += 1
            if time.time() - t0 >= budget_sec:
                break
        wall = max(0.2, time.time() - t0)
        video_sec = max(0.0, last_sec - first_sec) if first_sec is not None else 0.0
        res = None
        if video_sec > 1.0:
            res = {'rate': video_sec / wall,
                   'density': samples / video_sec}
        _PROBE_CACHE[key] = res
        return res
    except Exception:
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def process_video(input_paths, roi, callback, begin_attempt_number=1, params=None,
                  progress_callback=None, should_continue=None, slowmo_factor=1,
                  scan_mode=False, model=None, restrict_range=None,
                  log_fn=None, start_file_index=0, resume_pos=0.0):
    """Обрабатывает видео и вырезает попытки.

    callback(attempt: AttemptInfo) вызывается после обнаружения каждой попытки;
    если вернёт False - обработка останавливается.
    progress_callback(file_index, file_count, sample_index, sample_count) - прогресс.
    should_continue() - если вернёт False, обработка останавливается (кнопка "Остановить").
    slowmo_factor - коэффициент замедления записи (например, 4 для слоумо 96 к/с,
    записанного как 24 к/с). Все времена - в реальных секундах.
    scan_mode - режим сканирования: редкие сэмплы (шаг params['scan_interval'],
    точечный seek) и пониженный порог params['scan_threshold'];
    попытки возвращаются как кандидаты (AttemptInfo.is_candidate=True), без точных границ.
    model - предзагруженная модель YOLO (переиспользуется между вызовами).
    restrict_range - (start_real, end_real): анализировать только этот диапазон
    реального времени (для точной стадии ручного режима).
    Скан всегда идёт по ключевым кадрам (I-кадрам) стримингом: ffmpeg
    декодирует только I-кадры в пайп, нейросеть потребляет по мере
    поступления - без промежуточных файлов и ожидания полного прохода.
    Шаг сэмплирования - params['scan_interval'] (при редких I-кадрах -
    все I-кадры). Если I-кадры в среднем реже KEYFRAME_MAX_GAP сек -
    откат к обычным сэмплам. Только для scan_mode.
    log_fn - callback для лога (GUI-панель приложения); по умолчанию консоль.
    start_file_index/resume_pos - продолжение прерванного прогона: файлы с
    индексом меньше start_file_index пропускаются, в файле start_file_index
    анализ начинается с позиции resume_pos (реальные секунды; декодирование
    стрима стартует с ближайшего ключевого кадра перед позицией).
    """
    global _LOG_FN
    if log_fn is not None:
        _LOG_FN = log_fn
    if params is None:
        params = DEFAULT_PROCESSING_PARAMS.copy()
    min_strength = params.get('min_detection_strength', 0.5)
    if model is None:
        _log(i18n.tr("Загрузка нейросети YOLO (при первом запуске - до ~20с)..."))
        model = load_yolo_model()

    video_files = sorted(input_paths, key=os.path.basename)
    attempt_number = begin_attempt_number
    file_count = len(video_files)
    scan_interval = float(params.get('scan_interval', SCAN_INTERVAL))
    sample_rate = (1.0 / scan_interval) if scan_mode \
        else float(params.get('precise_analysis_fps', TARGET_ANALYSIS_FPS))
    range_start, range_end = restrict_range if restrict_range else (None, None)
    use_keyframes = scan_mode and KEYFRAME_SCAN

    # Prefetch следующего файла: медленное открытие MSMF (на Windows до
    # 10-30с на 4ГБ MTS с флешки) делается в фоне, пока YOLO сканирует
    # текущий файл. Ключевые кадры при стриминге готовить заранее не нужно.
    cap_prefetch = {}         # file_index -> (cap, elapsed)
    cap_prefetch_started = set()

    def _kf_prefetch_cleanup():
        for cap in cap_prefetch.values():
            if cap is not None:
                cap.release()
        cap_prefetch.clear()

    def _prefetch_worker(idx, path):
        try:
            t0 = time.time()
            cap = open_capture(path)
            cap_prefetch[idx] = (cap, time.time() - t0)
        except Exception:
            cap_prefetch[idx] = (None, 0.0)

    for file_index, video_file in enumerate(video_files):
        if file_index < start_file_index:
            continue
        _log(f"Processing {video_file}...")
        prefetched = cap_prefetch.pop(file_index, None) if scan_mode else None

        # Per-file диапазон и позиция резюме (только для стартового файла)
        file_range_start = range_start
        file_seek_pos = None
        if file_index == start_file_index and resume_pos > 0:
            file_range_start = resume_pos if range_start is None else max(range_start, resume_pos)
            # -ss ищет в контейнерном времени: для слоумо умножаем
            file_seek_pos = resume_pos * slowmo_factor if slowmo_factor > 1 else resume_pos

        # Сразу запускаем в фоне открытие следующего файла - оно перекроется
        # сканом текущего
        nxt = file_index + 1
        if scan_mode and nxt < file_count and nxt not in cap_prefetch_started:
            cap_prefetch_started.add(nxt)
            threading.Thread(target=_prefetch_worker, args=(nxt, video_files[nxt]),
                             daemon=True).start()

        if prefetched is not None:
            cap, open_elapsed = prefetched
            if cap is None or not cap.isOpened():
                _log(i18n.tr("  Не удалось открыть файл: {file}", file=video_file))
                continue
            _log(i18n.tr("  Файл открыт в фоне (за {sec:.1f}с)", sec=open_elapsed))
        else:
            open_t0 = time.time()
            cap = open_capture(video_file)
            if not cap.isOpened():
                _log(i18n.tr("  Не удалось открыть файл: {file}", file=video_file))
                continue
            _log(i18n.tr("  Файл открыт за {sec:.1f}с", sec=time.time() - open_t0))

        fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_real = fps_container * slowmo_factor if slowmo_factor > 1 else fps_container
        real_duration = total_frames / fps_real if fps_real > 0 else 0.0
        if slowmo_factor > 1:
            _log(i18n.tr("  Слоумо-видео (замедление 1/{factor}): "
                         "fps контейнера {container:.1f} -> реальный fps {real:.1f}",
                         factor=slowmo_factor, container=fps_container, real=fps_real))

        stream = None
        if use_keyframes:
            stream = _KeyframeStreamIter(cap, video_file, fps_real, total_frames,
                                         real_duration, scan_interval,
                                         slowmo_factor, file_range_start, range_end,
                                         sample_rate, seek_pos=file_seek_pos)
            sample_iter = iter(stream)
            analysis_rate = stream.rate  # уточнится после бутстрапа стрима
            sample_count = stream.sample_count
        elif scan_mode:
            # Скан: редкие сэмплы точечным seek - не декодируем весь файл
            sample_iter = _seek_sample_iter(cap, fps_real, total_frames, sample_rate,
                                            file_range_start, range_end)
            analysis_rate = sample_rate
            sample_count = int(real_duration * sample_rate) + 1
        else:
            # Полный декод с прореживанием (как раньше)
            frame_skip = max(1, int(round(fps_real / sample_rate)))
            analysis_rate = fps_real / frame_skip
            sample_iter = _grab_sample_iter(cap, fps_container, fps_real, total_frames,
                                            frame_skip, file_range_start, range_end)
            sample_count = total_frames

        detector = None
        if stream is None:
            detector = _Detector(model, roi, video_file,
                                 analysis_rate=analysis_rate,
                                 real_duration=real_duration,
                                 params=params, min_strength=min_strength,
                                 attempt_number=attempt_number, callback=callback,
                                 scan_mode=scan_mode,
                                 range_start=file_range_start, range_end=range_end)

        file_t0 = time.time()
        last_hb = file_t0
        sample_index = 0
        result = None
        stopped = False
        try:
            for real_sec, frame, pos, pos_total in sample_iter:
                if should_continue is not None and not should_continue():
                    _log(i18n.tr("  Остановка обработки по запросу пользователя"))
                    stopped = True
                    _kf_prefetch_cleanup()
                    return
                if detector is None:
                    # Стриминговый скан: rate известен только после бутстрапа
                    detector = _Detector(model, roi, video_file,
                                         analysis_rate=stream.rate,
                                         real_duration=real_duration,
                                         params=params, min_strength=min_strength,
                                         attempt_number=attempt_number,
                                         callback=callback, scan_mode=scan_mode,
                                         range_start=file_range_start,
                                         range_end=range_end)
                sample_index += 1
                if progress_callback is not None:
                    if stream is not None:
                        # Стриминг: точное число сэмплов заранее неизвестно
                        # (шаг квантуется по сетке I-кадров), поэтому прогресс
                        # - по позиции в файле (реальное время из pts)
                        progress_callback(file_index, file_count,
                                          int(real_sec * 100),
                                          int(max(real_duration, real_sec) * 100))
                    else:
                        progress_callback(file_index, file_count, sample_index,
                                          sample_count)
                now = time.time()
                if now - last_hb >= 30.0:
                    last_hb = now
                    _log(i18n.tr("  Прогресс: {sec:.0f}/{dur:.0f}с, {rate:.1f} сэмплов/с",
                                 sec=real_sec, dur=real_duration,
                                 rate=sample_index / max(1e-9, now - file_t0)))

                result = detector.process_sample(real_sec, frame)
                if result is False:
                    stopped = True
                    _kf_prefetch_cleanup()
                    return
            if stream is not None and progress_callback is not None:
                # Финальные проценты: последний сэмпл не доходит до конца файла
                progress_callback(file_index, file_count,
                                  int(real_duration * 100), int(real_duration * 100))
        finally:
            if stream is not None:
                try:
                    sample_iter.close()
                except Exception:
                    pass
            if not stopped:
                if detector is not None:
                    result = detector.finish()
                    attempt_number = detector.attempt_number
            cap.release()
        if result is False:
            _kf_prefetch_cleanup()
            return
    _kf_prefetch_cleanup()


# Атрибутика атлетов вынесена в athlete_attribution.py (алиас для
# совместимости: process_video_athletes и старый код ссылаются на имя)
from athlete_attribution import AthleteZoneProfiler as _AthleteZoneProfiler


def process_video_athletes(video_path, roi, should_continue=None,
                           progress_cb=None, log_cb=None, params=None,
                           attempt_found_cb=None, reid_embed_rate=5.0,
                           slowmo_factor=1.0, resume_pos=None,
                           athlete_attribution=True):
    """Единый быстрый пайплайн атлет-режимов (3/4): кейфрейм-скан зон тем же
    детектором, что и режимы 1/2, + накопление атрибутики атлета по сэмплам
    открытой зоны (vest/helmet HSV, reid OSNet с троттлингом, boat hue,
    thumb_t). athlete_attribution=False - только границы зон, без
    атрибутики (режим «Сохранение попыток» на текущем этапе). Возвращает
    список dict {start, end, thumb_t, vest, helmet, reid, boat, boat_top} -
    совместимо с результатами attempts_v2 (без tiles/helmet-банков
    сегментации); attempt_found_cb вызывается сразу при финализации зоны
    (прогрессивный UI)."""
    global _LOG_FN
    if log_cb is not None:
        _LOG_FN = log_cb
    if params is None:
        params = DEFAULT_PROCESSING_PARAMS.copy()
    p = params.copy()
    min_strength = p.get('min_detection_strength', 0.5)

    embedder = None
    if athlete_attribution:
        import attempts_v2
        try:
            import torch
            device = attempts_v2.default_device() if hasattr(attempts_v2, 'default_device') else \
                ('mps' if torch.backends.mps.is_available() else 'cpu')
            embedder = attempts_v2._get_embedder(device)
        except Exception as e:
            _log(i18n.tr("  Атрибутика недоступна ({err}) - зоны будут без reid", err=e))

    model = load_yolo_model()
    cap = open_capture(video_path)
    if not cap.isOpened():
        _log(i18n.tr("  Не удалось открыть файл: {file}", file=video_path))
        return []
    try:
        fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_real = fps_container * slowmo_factor if slowmo_factor > 1 else fps_container
        real_duration = total_frames / fps_real if fps_real > 0 else 0.0
        scan_interval = float(p.get('scan_interval', SCAN_INTERVAL))
        seek = resume_pos * slowmo_factor if slowmo_factor > 1 and resume_pos else resume_pos
        stream = _KeyframeStreamIter(cap, video_path, fps_real, total_frames,
                                     real_duration, scan_interval, slowmo_factor,
                                     resume_pos, None, 1.0 / scan_interval,
                                     seek_pos=seek)
        # Детектор создаём лениво на первом сэмпле (как в process_video):
        # к тому моменту бутстрап стрима уже уточнит stream.rate - иначе
        # pause_samples считается по грубой оценке 1/scan_interval и зоны
        # закрываются дольше, чем в остальных режимах (слипание попыток)
        detector = None
        # ROI в координатах сэмпла (600px) - для thumb и profiler.
        # Детектору ROI передаётся как есть (полигон или прям) - идентично
        # process_video; bbox нужен только профайлеру атрибутики
        if roi and not (len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi)):
            xs = [p[0] for p in roi]
            ys = [p[1] for p in roi]
            roi_bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            roi_bbox = roi
        results_out = []
        cur_prof = [None]
        model_names = model.names

        def _collect(info):
            """Callback детектора: зона финализирована - отправляем результат."""
            prof = cur_prof[0]
            cur_prof[0] = None
            if info is None:
                return True
            if prof is None:
                if not athlete_attribution:
                    # только границы: минимальный результат без атрибутики
                    r = {'start': info.start, 'end': info.end,
                         'thumb_t': info.thumb_t if info.thumb_t is not None else info.start,
                         'thumb_frame': info.thumb_frame,
                         'vest': None,
                         'helmet': None, 'reid': None, 'boat': None,
                         'boat_top': None}
                    results_out.append(r)
                    if attempt_found_cb is not None:
                        return attempt_found_cb(r) is not False
                return True
            r = prof.result(info.start, info.end)
            results_out.append(r)
            cur_prof[0] = None
            if attempt_found_cb is not None:
                return attempt_found_cb(r) is not False
            return True

        sample_iter = iter(stream)
        try:
            _perf_t = time.time()
            for real_sec, frame, pos, pos_total in sample_iter:
                if should_continue is not None and not should_continue():
                    _log(i18n.tr("  Остановка обработки по запросу пользователя"))
                    return results_out
                if detector is None:
                    detector = _Detector(model, roi, video_path,
                                         analysis_rate=stream.rate,
                                         real_duration=real_duration,
                                         params=p, min_strength=min_strength,
                                         attempt_number=1,
                                         callback=_collect,
                                         scan_mode=True)
                _now = time.time()
                if _now - _perf_t > 0.5:
                    _log(i18n.tr("  [perf] сэмпл @{t:.1f}с: ожидание {dt:.1f}с "
                                 "(зона={zone})",
                                 t=real_sec, dt=_now - _perf_t,
                                 zone='да' if detector.in_attempt else 'нет'))
                _perf_t = _now
                det_res = detector.process_sample(real_sec, frame)
                _now2 = time.time()
                if _now2 - _perf_t > 0.5:
                    _log(i18n.tr("  [perf] обработка сэмпла @{t:.1f}с: {dt:.1f}с",
                                 t=real_sec, dt=_now2 - _perf_t))
                    _perf_t = _now2
                # накопление профиля открытой зоны (без 2го инференса YOLO -
                # берём результаты детектора); при athlete_attribution=False
                # профили не копятся вовсе
                if athlete_attribution and detector.in_attempt \
                        and detector.start_time is not None:
                    if cur_prof[0] is None or cur_prof[0].start != detector.start_time:
                        cur_prof[0] = _AthleteZoneProfiler(
                            detector.start_time, roi_bbox,
                            reid_embed_rate=reid_embed_rate, embedder=embedder)
                    cur_prof[0]._model_names = model_names
                    cur_prof[0].on_sample(real_sec, frame,
                                          getattr(detector, 'last_results', None))
                if det_res is False:
                    return results_out
                if progress_cb is not None:
                    progress_cb(int(real_sec * 100),
                                int(max(real_duration, real_sec) * 100))
            if detector is not None:
                fin = detector.finish()
                if fin is not None:
                    _collect(fin)
        finally:
            try:
                sample_iter.close()
            except Exception:
                pass
    finally:
        cap.release()
    return results_out
