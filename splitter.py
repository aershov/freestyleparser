import cv2
import glob
import i18n
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
                 is_candidate=False):
        self.source_video = source_video
        self.start = start
        self.end = end
        self.number = number
        self.best_frame = best_frame
        self.person_frame = person_frame
        self.base_frame = base_frame
        self.person_bbox = person_bbox
        self.is_candidate = is_candidate

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


def load_yolo_model():
    """Загружает модель YOLO с кешем: загрузка занимает секунды,
    переиспользуем между файлами/потоками (модель stateless)."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/yolo11s-seg.pt')
        _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


def _finalize_attempt(attempt_number, start_time, end_time, real_duration,
                      person_bbox, base_frame, best_frame, person_frame,
                      fallback_frame, video_file, callback, params, scan_mode=False):
    """Создаёт попытку и вызывает callback. Возвращает (attempt, attempt_number) или (None, attempt_number) или (False, attempt_number) для стопа."""
    start_padding = params['attempt_start_padding']
    end_padding = params['attempt_end_padding']
    min_duration = params['min_attempt_duration']

    start1 = max(0, start_time - start_padding)
    end1 = min(end_time + end_padding, real_duration)

    if end1 <= start1:
        _log(i18n.tr("  [SKIP] Попытка #{n}: start ({start:.2f}s) >= end ({end:.2f}s)",
                           n=attempt_number, start=start1, end=end1))
        return None, attempt_number

    duration = end1 - start1
    if duration < min_duration:
        _log(i18n.tr("  [SKIP] Попытка #{n}: duration {dur:.2f}s < {min}s",
                           n=attempt_number, dur=duration, min=min_duration))
        return None, attempt_number

    if person_bbox is None:
        _log(i18n.tr("  [SKIP] Попытка #{n}: нет данных о позиции человека", n=attempt_number))
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
                          is_candidate=scan_mode)
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


def _keyframe_sample_iter(kf_samples, slowmo_factor, step, range_start, range_end):
    """Генератор сэмплов из заранее извлечённых опорных кадров.
    kf_samples - [(file_time, path)]; берётся каждый step-й кадр."""
    if not kf_samples:
        return
    for i, (ftime, path) in enumerate(kf_samples):
        if i % step:
            continue
        real_sec = ftime / slowmo_factor if slowmo_factor > 1 else ftime
        if range_start and real_sec < range_start:
            continue
        if range_end is not None and real_sec > range_end:
            break
        frame = cv2.imread(path)
        if frame is None:
            continue
        yield real_sec, frame, i + 1, len(kf_samples)


def _extract_keyframes(video_file, tmp_dir):
    """Рендерит ключевые кадры видео в jpg (ffmpeg -skip_frame nokey: декодер
    получает только I-кадры - гарантированно минимальное декодирование).
    Метки времени берутся из showinfo того же прохода.
    Возвращает ([(pts, path)], median_gap) или (None, 0) при сбое."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None, 0.0
    pattern = os.path.join(tmp_dir, 'kf_%06d.jpg')
    cmd = [ffmpeg, '-hide_banner', '-skip_frame', 'nokey', '-i', video_file,
           '-vf', f'showinfo,scale={YOLO_MAX_WIDTH}:-2', '-vsync', '0',
           '-q:v', '3', '-y', pattern]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        if proc.returncode != 0:
            return None, 0.0
        out = proc.stderr.decode('utf-8', 'replace')
    except Exception:
        return None, 0.0
    pts = []
    for m in re.finditer(r'pts_time:([0-9.]+)', out):
        try:
            pts.append(float(m.group(1)))
        except ValueError:
            continue
    files = sorted(glob.glob(os.path.join(tmp_dir, 'kf_*.jpg')))
    if not files:
        return None, 0.0
    if len(pts) > len(files):
        pts = pts[:len(files)]
    while len(pts) < len(files):
        pts.append(pts[-1] if pts else 0.0)
    gaps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2] if gaps else 0.0
    return list(zip(pts, files)), median_gap


def _median_gap(pts):
    """Медианный интервал между метками времени (с)."""
    gaps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _select_keyframes(kf_all, scan_interval):
    """Выбирает из всех I-кадров сэмплы с шагом ~scan_interval. I-кадр
    пропускается, только если накопленная пауза почти достигла шага скана
    (допуск 0.5*GOP): иначе при GOP чуть меньше шага (например 1.99с при
    шаге 2с) пропуск раздваивал бы эффективный шаг до ~4с, а при плотном
    GOP квантование по сетке I-кадров съезжало бы вниз (0.78с вместо 1с).
    Итоговый шаг лежит в интервале [шаг - 0.5*GOP; шаг + 0.5*GOP)."""
    if not kf_all:
        return []
    gop = _median_gap([pts for pts, _ in kf_all])
    threshold = max(0.0, scan_interval - gop * 0.5)
    selected = []
    last_t = None
    for pts, path in kf_all:
        if last_t is None or pts - last_t >= threshold - 1e-6:
            selected.append((pts, path))
            last_t = pts
    return selected


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
            scan_mode=self.scan_mode)
        if result is not None:
            self.best_frame = None
            self.base_frame = None
            self.person_bbox = None
        return result

    def finish(self):
        """Финализирует незакрытую попытку на конце файла/диапазона."""
        if self.in_attempt and self.start_time is not None:
            end_time = self.real_duration if self.range_end is None else min(self.real_duration, self.range_end)
            _log(i18n.tr("  [{file}] Финализация попытки #{n} на конце ({t:.1f}s)",
                               file=os.path.basename(self.video_file),
                               n=self.attempt_number, t=end_time))
            return self._finalize(self.start_time, end_time)
        return None

def process_video(input_paths, roi, callback, begin_attempt_number=1, params=None,
                  progress_callback=None, should_continue=None, slowmo_factor=1,
                  scan_mode=False, model=None, restrict_range=None,
                  log_fn=None):
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
    Скан всегда идёт по ключевым кадрам (I-кадрам): декодируется минимум,
    быстро с флешки. Сэмплы выбираются из I-кадров с шагом params['scan_interval']
    (реже шага - все I-кадры). Если I-кадры в среднем реже KEYFRAME_MAX_GAP сек -
    откат к обычным сэмплам. Только для scan_mode.
    log_fn - callback для лога (GUI-панель приложения); по умолчанию консоль.
    """
    global _LOG_FN
    if log_fn is not None:
        _LOG_FN = log_fn
    if params is None:
        params = DEFAULT_PROCESSING_PARAMS.copy()
    min_strength = params.get('min_detection_strength', 0.5)
    if model is None:
        model = load_yolo_model()

    video_files = sorted(input_paths, key=os.path.basename)
    attempt_number = begin_attempt_number
    file_count = len(video_files)
    scan_interval = float(params.get('scan_interval', SCAN_INTERVAL))
    sample_rate = (1.0 / scan_interval) if scan_mode else TARGET_ANALYSIS_FPS
    range_start, range_end = restrict_range if restrict_range else (None, None)
    use_keyframes = scan_mode and KEYFRAME_SCAN

    for file_index, video_file in enumerate(video_files):
        _log(f"Processing {video_file}...")
        cap = open_capture(video_file)
        if not cap.isOpened():
            _log(i18n.tr("  Не удалось открыть файл: {file}", file=video_file))
            continue

        fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_real = fps_container * slowmo_factor if slowmo_factor > 1 else fps_container
        real_duration = total_frames / fps_real if fps_real > 0 else 0.0
        if slowmo_factor > 1:
            _log(i18n.tr("  Слоумо-видео (замедление 1/{factor}): "
                         "fps контейнера {container:.1f} -> реальный fps {real:.1f}",
                         factor=slowmo_factor, container=fps_container, real=fps_real))

        kf_samples = None
        kf_gap = 0.0
        tmp_dir = None
        if use_keyframes:
            tmp_dir = tempfile.mkdtemp(prefix='FreestyleParserKF_')
            kf_all, kf_gap = _extract_keyframes(video_file, tmp_dir)
            if kf_all is None:
                _log(i18n.tr("  Не удалось извлечь ключевые кадры - скан обычными сэмплами"))
            elif kf_gap > KEYFRAME_MAX_GAP:
                _log(i18n.tr("  Ключевые кадры редкие (шаг ~{gap:.1f}с > {max:.0f}с) - "
                             "скан обычными сэмплами",
                             gap=kf_gap, max=KEYFRAME_MAX_GAP))
            else:
                kf_samples = _select_keyframes(kf_all, scan_interval)
                sel_gap = _median_gap([pts for pts, _ in kf_samples])
                _log(i18n.tr("  Скан по ключевым кадрам: {n} сэмплов, шаг ~{step:.2f}с "
                             "(I-кадры в файле каждые {gop:.2f}с)",
                             n=len(kf_samples), step=sel_gap, gop=kf_gap))
                if sel_gap > scan_interval + 1e-6:
                    _log(i18n.tr("  Внимание: I-кадры ({gop:.2f}с) реже шага скана "
                                 "({si:.1f}с) - детекция грубее, возможны пропуски",
                                 gop=kf_gap, si=scan_interval))
                    _log(i18n.tr("  Рекомендация: шаг скана {step} для этого файла "
                                 "(I-кадры каждые {gop:.2f}с)",
                                 step=_recommended_scan_step(kf_gap), gop=kf_gap))
                kf_gap = sel_gap
            if kf_samples is None and tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir = None

        if kf_samples:
            sample_iter = _keyframe_sample_iter(kf_samples, slowmo_factor, 1,
                                                range_start, range_end)
            analysis_rate = 1.0 / kf_gap if kf_gap > 0 else sample_rate
            sample_count = len(kf_samples)
        elif scan_mode:
            # Скан: редкие сэмплы точечным seek - не декодируем весь файл
            sample_iter = _seek_sample_iter(cap, fps_real, total_frames, sample_rate,
                                            range_start, range_end)
            analysis_rate = sample_rate
            sample_count = int(real_duration * sample_rate) + 1
        else:
            # Полный декод с прореживанием (как раньше)
            frame_skip = max(1, int(round(fps_real / sample_rate)))
            analysis_rate = fps_real / frame_skip
            sample_iter = _grab_sample_iter(cap, fps_container, fps_real, total_frames,
                                            frame_skip, range_start, range_end)
            sample_count = total_frames

        detector = _Detector(model, roi, video_file,
                             analysis_rate=analysis_rate,
                             real_duration=real_duration,
                             params=params, min_strength=min_strength,
                             attempt_number=attempt_number, callback=callback,
                             scan_mode=scan_mode,
                             range_start=range_start, range_end=range_end)

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
                    return
                sample_index += 1
                if progress_callback is not None:
                    progress_callback(file_index, file_count, sample_index, sample_count)
                now = time.time()
                if now - last_hb >= 30.0:
                    last_hb = now
                    _log(i18n.tr("  Прогресс: {sec:.0f}/{dur:.0f}с, {rate:.1f} сэмплов/с",
                                 sec=real_sec, dur=real_duration,
                                 rate=sample_index / max(1e-9, now - file_t0)))

                result = detector.process_sample(real_sec, frame)
                if result is False:
                    stopped = True
                    return
        finally:
            if not stopped:
                result = detector.finish()
                attempt_number = detector.attempt_number
            cap.release()
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        if result is False:
            return
            sample_index += 1
            if progress_callback is not None:
                progress_callback(file_index, file_count, sample_index, sample_count)

            result = detector.process_sample(real_sec, frame)
            if result is False:
                cap.release()
                return

        result = detector.finish()
        attempt_number = detector.attempt_number
        cap.release()
        if result is False:
            return
