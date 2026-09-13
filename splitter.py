import cv2
import numpy as np
import os
import sys
from collections import deque
from ultralytics import YOLO

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Порог уверенности для детекции объектов
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 600  # уменьшаем кадр перед анализом в YOLO до стольких пикселей

# Конфигурация обработки видео
TARGET_ANALYSIS_FPS = 5  # Целевая частота анализа (сэмплов в секунду реального времени)
MIN_DETECTION_TIME = 0.6  # Минимальное время (в секундах) с детекциями для начала попытки

# Режим сканирования (ручной выбор попыток)
SCAN_INTERVAL = 2.5     # Шаг сканирования, секунд реального времени между сэмплами
SCAN_CONFIDENCE = 0.25  # Пониженный порог YOLO для сканирования

DEFAULT_PROCESSING_PARAMS = {
    'min_attempt_duration': 5,     # Минимальная длительность попытки в секундах
    'min_pause_duration': 3,       # Минимальная пауза между попытками в секундах
    'attempt_start_padding': 2,    # Запас времени (в секундах) к началу попытки
    'attempt_end_padding': 0.5,    # Запас времени (в секундах) от конца попытки
    'min_detection_strength': 0.5, # Минимальная сила сигнала (0-1) для начала/продолжения попытки
}

np.seterr(divide='ignore', invalid='ignore')

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


def load_yolo_model():
    """Загружает модель YOLO (вынесено наружу, чтобы переиспользовать между вызовами)"""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/yolo11s-seg.pt')
    return YOLO(model_path)


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
        safe_print(f"  [SKIP] Попытка #{attempt_number}: start ({start1:.2f}s) >= end ({end1:.2f}s)")
        return None, attempt_number

    duration = end1 - start1
    if duration < min_duration:
        safe_print(f"  [SKIP] Попытка #{attempt_number}: duration {duration:.2f}s < {min_duration}s")
        return None, attempt_number

    if person_bbox is None:
        safe_print(f"  [SKIP] Попытка #{attempt_number}: нет данных о позиции человека")
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
    attempts exceeded"), поэтому сначала пробуем Media Foundation."""
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(video_file, cv2.CAP_MSMF)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(video_file)


def _seek_sample_iter(cap, fps_real, total_frames, sample_rate, range_start, range_end):
    """Генератор сэмплов точечными seek'ами. Для редкой выборки (скан) намного
    быстрее полного декодирования: каждый seek декодирует не больше кадров,
    чем GOP. sample_rate - сэмплов в секунду реального времени."""
    step = max(1, int(round(fps_real / sample_rate)))
    safe_print(f"  Скан точечным seek: шаг {step} кадров, ~{sample_rate:.1f}/с реальных")
    start_f = int(range_start * fps_real) if range_start and range_start > 0 else 0
    end_f = total_frames if range_end is None else min(total_frames, int(range_end * fps_real))
    pos = start_f
    while pos < end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break
        yield pos / fps_real, frame, pos, end_f
        pos += step

def _grab_sample_iter(cap, fps_container, fps_real, total_frames, frame_skip,
                      range_start, range_end):
    """Генератор сэмплов из полного декодирования (как раньше).
    Декодируется каждый frame_skip-й кадр; сэмплы в реальных секундах."""
    safe_print(f"  FPS: {fps_container:.1f}, реальный fps: {fps_real:.1f}, "
               f"frame_skip: {frame_skip}, анализ: {fps_real / frame_skip:.1f}/с")
    start_frame = 0
    if range_start and range_start > 0:
        start_frame = int(range_start * fps_real)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            safe_print(f"  Старт с кадра {start_frame} (диапазон от {range_start:.1f}s)")
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

        self.conf_threshold = SCAN_CONFIDENCE if scan_mode else YOLO_CONFIDENCE_THRESHOLD
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
            if len(self.roi) == 4:  # Прямоугольник
                left, top, right, bottom = self.roi
                x1_px = int(frame_width * left / 100)
                y1_px = int(frame_height * top / 100)
                x2_px = int(frame_width * right / 100)
                y2_px = int(frame_height * bottom / 100)
                frame = frame[y1_px:y2_px, x1_px:x2_px]
                original_frame = original_frame[y1_px:y2_px, x1_px:x2_px]
            elif len(self.roi) > 4:  # Многоугольник
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
                        safe_print(f"  [START] Попытка #{self.attempt_number} на {self.start_time:.2f}s")
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
            safe_print(f"  [{os.path.basename(self.video_file)}] Финализация попытки #{self.attempt_number} на конце ({end_time:.1f}s)")
            return self._finalize(self.start_time, end_time)
        return None

def process_video(input_paths, roi, callback, begin_attempt_number=1, params=None,
                  progress_callback=None, should_continue=None, slowmo_factor=1,
                  scan_mode=False, model=None, restrict_range=None):
    """Обрабатывает видео и вырезает попытки.

    callback(attempt: AttemptInfo) вызывается после обнаружения каждой попытки;
    если вернёт False - обработка останавливается.
    progress_callback(file_index, file_count, sample_index, sample_count) - прогресс.
    should_continue() - если вернёт False, обработка останавливается (кнопка "Остановить").
    slowmo_factor - коэффициент замедления записи (например, 4 для слоумо 96 к/с,
    записанного как 24 к/с). Все времена - в реальных секундах.
    scan_mode - режим сканирования: редкие сэмплы (SCAN_INTERVAL, точечный seek)
    и пониженный порог; попытки возвращаются как кандидаты
    (AttemptInfo.is_candidate=True), без точных границ.
    model - предзагруженная модель YOLO (переиспользуется между вызовами).
    restrict_range - (start_real, end_real): анализировать только этот диапазон
    реального времени (для точной стадии ручного режима).
    """
    if params is None:
        params = DEFAULT_PROCESSING_PARAMS.copy()
    min_strength = params.get('min_detection_strength', 0.5)
    if model is None:
        model = load_yolo_model()

    video_files = sorted(input_paths, key=os.path.basename)
    attempt_number = begin_attempt_number
    file_count = len(video_files)
    sample_rate = (1.0 / SCAN_INTERVAL) if scan_mode else TARGET_ANALYSIS_FPS
    range_start, range_end = restrict_range if restrict_range else (None, None)

    for file_index, video_file in enumerate(video_files):
        safe_print(f"Processing {video_file}...")
        cap = open_capture(video_file)
        if not cap.isOpened():
            safe_print(f"  Не удалось открыть файл: {video_file}")
            continue

        fps_container = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_real = fps_container * slowmo_factor if slowmo_factor > 1 else fps_container
        real_duration = total_frames / fps_real if fps_real > 0 else 0.0
        if slowmo_factor > 1:
            safe_print(f"  Слоумо-видео (замедление 1/{slowmo_factor}): "
                       f"fps контейнера {fps_container:.1f} -> реальный fps {fps_real:.1f}")

        if scan_mode:
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

        sample_index = 0
        for real_sec, frame, pos, pos_total in sample_iter:
            if should_continue is not None and not should_continue():
                safe_print("  Остановка обработки по запросу пользователя")
                cap.release()
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
