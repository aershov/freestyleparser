import cv2
import numpy as np
import os
import sys
from collections import deque
from ultralytics import YOLO

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Порог уверенности для детекции объектов (снижен с 0.6)
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 600  # уменьшаем кадр перед анализом в YOLO до стольких пикселей

# Конфигурация обработки видео
TARGET_ANALYSIS_FPS = 5  # Целевая частота анализа (кадров в секунду), не зависит от fps видео
MIN_DETECTION_TIME = 0.6  # Минимальное время (в секундах) с детекциями для начала попытки (снижен с 1.0)

DEFAULT_PROCESSING_PARAMS = {
    'min_attempt_duration': 5,     # Минимальная длительность попытки в секундах
    'min_pause_duration': 3,       # Минимальная пауза между попытками в секундах
    'attempt_start_padding': 2,    # Запас времени (в секундах) к началу попытки
    'attempt_end_padding': 0.5,    # Запас времени (в секундах) от конца попытки
    'min_detection_strength': 0.5, # Минимальная сила сигнала (0-1) для начала/продолжения попытки
}

np.seterr(divide='ignore', invalid='ignore')

class AttemptInfo:

    def __init__(self, source_video, start, end, number, best_frame, person_frame, base_frame, person_bbox):
        self.source_video = source_video
        self.start = start
        self.end = end
        self.number = number
        self.best_frame = best_frame
        self.person_frame = person_frame
        self.base_frame = base_frame
        self.person_bbox = person_bbox

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


def _finalize_attempt(attempt_number, start_time, end_time, total_frames, fps,
                      person_bbox, base_frame, best_frame, person_frame,
                      fallback_frame, video_file, callback, params):
    """Создаёт попытку и вызывает callback. Возвращает (attempt, attempt_number) или (None, attempt_number) или (False, attempt_number) для стопа."""
    start_padding = params['attempt_start_padding']
    end_padding = params['attempt_end_padding']
    min_duration = params['min_attempt_duration']

    start1 = max(0, start_time - start_padding)
    end1 = min(end_time + end_padding, total_frames / fps)

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
                          start=start1, end=end1)
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


def process_video(input_paths, roi, callback, begin_attempt_number=1, params=None,
                  progress_callback=None, should_continue=None):
    """Обрабатывает видео и вырезает попытки.

    callback(attempt: AttemptInfo) вызывается после обнаружения каждой попытки;
    если вернёт False - обработка останавливается.
    progress_callback(file_index, file_count, frame_index, total_frames) - прогресс обработки.
    should_continue() - если вернёт False, обработка останавливается (кнопка "Остановить").
    """
    if params is None:
        params = DEFAULT_PROCESSING_PARAMS.copy()

    min_pause_duration = params['min_pause_duration']
    min_strength = params.get('min_detection_strength', 0.5)

    # Загрузка модели YOLO
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '', 'models/yolo11s-seg.pt')
    model = YOLO(model_path)

    video_files = input_paths
    attempt_number = begin_attempt_number
    # Сортировка файлов по дате создания
    # video_files.sort(key=os.path.getctime)
    video_files.sort(key=os.path.basename, reverse=False)

    # Создание выходной папки, если она не существует
    # os.makedirs(output_folder, exist_ok=True)

    for file_index, video_file in enumerate(video_files):
        safe_print(f"Processing {video_file}...")
        cap = open_capture(video_file)
        if not cap.isOpened():
            safe_print(f"  Не удалось открыть файл: {video_file}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_limit = total_frames
        frame_skip = max(1, int(round(fps / TARGET_ANALYSIS_FPS)))
        safe_print(f"  FPS: {fps}, frame_skip: {frame_skip}, analysis rate: {fps / frame_skip:.1f} fps")

        in_attempt = False
        start_time = None
        end_time = None
        frames_in_attempt = 0
        frames_out_of_attempt = 0

        best_frame = None
        best_frame_confidence = 0
        person_frame = None
        base_frame = None
        person_bbox = None
        person_conf = 0
        last_person_bbox = None  # для fallback bbox когда person пропадает в брызгах
        small_frame_original = None

        detection_window_size = max(1, int(MIN_DETECTION_TIME * TARGET_ANALYSIS_FPS))
        detection_window = deque(maxlen=detection_window_size)
        bbox_history = deque(maxlen=detection_window_size)  # [(frame_num, x1,y1,x2,y2, conf), ...]

        frame_index = 0  # номер последнего прочитанного кадра (1-based, как CAP_PROP_POS_FRAMES)
        while True:
            if should_continue is not None and not should_continue():
                safe_print("  Остановка обработки по запросу пользователя")
                cap.release()
                return

            # grab() не декодирует кадр - намного быстрее read() для пропускаемых кадров
            if not cap.grab():
                break
            frame_index += 1
            if frame_index >= frame_limit:
                break

            current_frame_num = frame_index
            if current_frame_num % frame_skip == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                # Время считаем по счётчику кадров: CAP_PROP_POS_MSEC ненадёжен у некоторых бэкендов
                current_sec = frame_index / fps
                if progress_callback is not None:
                    progress_callback(file_index, len(video_files), frame_index, total_frames)
                if in_attempt and current_frame_num % (frame_skip * 50) == 0:
                    safe_print(f"  [{os.path.basename(video_file)}] frame {current_frame_num}/{total_frames}, {current_sec:.1f}s, in attempt #{attempt_number}")
                elif current_frame_num % (frame_skip * 100) == 0:
                    safe_print(f"  [{os.path.basename(video_file)}] frame {current_frame_num}/{total_frames}, {current_sec:.1f}s")

                original_frame = frame.copy()
                
                # Установка области интереса
                if roi:
                    frame_height, frame_width = frame.shape[:2]
                    
                    if len(roi) == 4:  # Прямоугольник
                        left, top, right, bottom = roi
                        x1_px = int(frame_width * left / 100)
                        y1_px = int(frame_height * top / 100)
                        x2_px = int(frame_width * right / 100)
                        y2_px = int(frame_height * bottom / 100)
                        frame = frame[y1_px:y2_px, x1_px:x2_px]
                        original_frame = original_frame[y1_px:y2_px, x1_px:x2_px]
                    elif len(roi) > 4:  # Многоугольник
                        polygon_points = [(int(frame_width * x / 100), int(frame_height * y / 100)) for x, y in roi]
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

                results = model(small_frame, verbose=False)

                # Проверка наличия объектов в ТЕКУЩЕМ кадре
                person_detected_in_frame = False
                best_conf_in_frame = 0
                best_box_in_frame = None
                best_person_box = None
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls]
                        if class_name in YOLO_TARGET_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            person_detected_in_frame = True
                            if conf > best_conf_in_frame:
                                best_conf_in_frame = conf
                                best_box_in_frame = box
                        if class_name == 'person' and conf > person_conf:
                            best_person_box = box
                            person_conf = conf

                # Обновляем лучший bbox для превью и историю
                best_box_np = None
                if best_box_in_frame is not None:
                    best_box_np = best_box_in_frame.xyxy[0].cpu().numpy()
                    bx1, by1, bx2, by2 = map(int, best_box_np)

                    # Отслеживаем person bbox отдельно (fallback для превью)
                    if best_person_box is not None:
                        last_person_bbox = best_person_box.xyxy[0].cpu().numpy()
                        best_person_box_np = last_person_bbox
                    else:
                        best_person_box_np = last_person_bbox if last_person_bbox is not None else best_box_np

                    person_bbox = best_person_box_np
                    bbox_history.append((current_frame_num, bx1, by1, bx2, by2, best_conf_in_frame))

                    # Обновляем лучший кадр для превью (приоритет к person)
                    if best_person_box is not None:
                        ppx1, ppy1, ppx2, ppy2 = map(int, best_person_box_np)
                        person_frame = small_frame_original[ppy1:ppy2, ppx1:ppx2]
                    else:
                        person_frame = small_frame_original[by1:by2, bx1:bx2]

                    if best_conf_in_frame > best_frame_confidence:
                        best_frame_confidence = best_conf_in_frame
                        best_frame = small_frame_original

                    if person_conf > 0.5:
                        best_frame_confidence = person_conf
                        best_frame = small_frame_original

                elif best_frame is None and small_frame_original is not None:
                    best_frame = small_frame_original
                    person_frame = small_frame_original

                # Вычисляем detection strength: bbox motion (IoU) + confidence drift
                detection_strength = 0.0
                if person_detected_in_frame and len(bbox_history) >= 2:
                    ious = []
                    confs = []
                    for i in range(1, len(bbox_history)):
                        prev = bbox_history[i - 1]
                        curr = bbox_history[i]
                        ious.append(_iou(prev[1:5], curr[1:5]))
                        confs.append(abs(curr[5] - prev[5]))
                    avg_iou = sum(ious) / len(ious)
                    avg_conf_drift = sum(confs) / len(confs)
                    detection_strength = (1.0 - avg_iou) * 0.7 + min(avg_conf_drift * 10, 1.0) * 0.3

                # Hysteresis: во время попытки порог strength ниже (ловим брызги/переходные моменты)
                actual_min_strength = min_strength * 0.3 if in_attempt else min_strength
                has_detections = person_detected_in_frame and detection_strength >= actual_min_strength

                # запоминаем базовый фрейм без человека
                if not in_attempt:
                    if base_frame is None:
                        base_frame = small_frame_original

                if has_detections:
                    detection_window.append(1)
                    frames_in_attempt += 1
                    frames_out_of_attempt = 0
                    if not in_attempt:
                        # Debounce: требуем стабильных детекций перед стартом попытки
                        if len(detection_window) >= detection_window_size and all(detection_window):
                            detection_start_time = current_sec
                            if detection_start_time - (end_time or 0) >= min_pause_duration:
                                in_attempt = True
                                start_time = detection_start_time
                                safe_print(f"  [START] Попытка #{attempt_number} на {start_time:.2f}s")
                else:
                    detection_window.append(0)
                    frames_out_of_attempt += 1
                    if in_attempt and frames_out_of_attempt >= min_pause_duration * fps / frame_skip:
                        in_attempt = False
                        best_frame_confidence = 0
                        detection_window.clear()
                        bbox_history.clear()
                        person_conf = 0
                        last_person_bbox = None
                        end_time = current_sec

                        result, attempt_number = _finalize_attempt(
                            attempt_number, start_time, end_time, total_frames, fps,
                            person_bbox, base_frame, best_frame, person_frame,
                            small_frame_original, video_file, callback, params)
                        if result is not None:
                            best_frame = None
                            base_frame = None
                            person_bbox = None
                        if result is False:
                            cap.release()
                            return

        if in_attempt and start_time is not None:
            end_time = total_frames / fps
            safe_print(f"  [{os.path.basename(video_file)}] Финализация попытки #{attempt_number} на конце видео ({end_time:.1f}s)")
            result, attempt_number = _finalize_attempt(
                attempt_number, start_time, end_time, total_frames, fps,
                person_bbox, base_frame, best_frame, person_frame,
                small_frame_original, video_file, callback, params)

        if total_frames and frame_index < total_frames * 0.95:
            safe_print(f"  ВНИМАНИЕ: прочитано {frame_index} кадров из {total_frames} "
                       f"заявленных - файл, возможно, обработан не полностью")

        cap.release()
