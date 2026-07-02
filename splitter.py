import cv2
import numpy as np
import os
from ultralytics import YOLO

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.6  # Порог уверенности для детекции объектов
# YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_TARGET_CLASSES = ['person']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 600  # уменьшаем кадр перед анализом в YOLO до стольких пикселей


# Конфигурация обработки видео
FRAME_SKIP = 10  # Обрабатываем каждый N-й кадр //TODO надо поменять на указания N кадров в секунду анализа, чтобы не зависеть от частоты кадров в видео
MIN_DETECTION_TIME = 1  # Минимальное время (в секундах) с детекциями для начала попытки
MIN_ATTEMPT_DURATION = 5  # Минимальная длительность попытки в секундах
MIN_PAUSE_DURATION = 3  # Минимальная пауза между попытками в секундах (если потеряли атлета меньше чем на столько сек, то попытка продолжается, если больше - то новая попытка)

ATTEMPT_START_PADDING = 2  # Запас времени (в секундах) к началу попытки
ATTEMPT_END_PADDING = 0.5 # Запас времени (в секундах) от конца попытки

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


def _finalize_attempt(attempt_number, start_time, end_time, total_frames, fps,
                      person_bbox, base_frame, best_frame, person_frame,
                      fallback_frame, video_file, callback):
    """Создаёт попытку и вызывает callback. Возвращает (attempt, attempt_number) или (None, attempt_number) или (False, attempt_number) для стопа."""
    start1 = max(0, start_time - ATTEMPT_START_PADDING)
    end1 = min(end_time + ATTEMPT_END_PADDING, total_frames / fps)

    if end1 <= start1:
        safe_print(f"  [SKIP] Попытка #{attempt_number}: start ({start1:.2f}s) >= end ({end1:.2f}s)")
        return None, attempt_number

    duration = end1 - start1
    if duration < MIN_ATTEMPT_DURATION:
        safe_print(f"  [SKIP] Попытка #{attempt_number}: duration {duration:.2f}s < {MIN_ATTEMPT_DURATION}s")
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


def process_video(input_paths, roi, callback, begin_attempt_number=1):
    # Загрузка модели YOLO
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '', 'models/yolo11s-seg.pt')
    # if not os.path.exists(model_path):
    #     raise FileError("'yolov8n.pt' not found")
    model = YOLO(model_path)

    video_files = input_paths
    attempt_number = begin_attempt_number
    # Сортировка файлов по дате создания
    # video_files.sort(key=os.path.getctime)
    video_files.sort(key=os.path.basename, reverse=False)

    # Создание выходной папки, если она не существует
    # os.makedirs(output_folder, exist_ok=True)

    for video_file in video_files:
        safe_print(f"Processing {video_file}...")
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_limit = total_frames

        in_attempt = False
        start_time = None
        end_time = None
        frames_in_attempt = 0
        frames_out_of_attempt = 0

        best_frame = None # запоминаем кадр с максимально видимым человеком
        best_frame_confidence = 0 # confindence этого кадра
        person_frame = None # подкадр с человеком
        base_frame = None # первый кадр попытки
        person_bbox = None
        current_frame_small = None
        best_frame_original = None  # Оригинальный кадр без маски для превью
        small_frame_original = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_limit):
                break

            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_num % FRAME_SKIP == 0:
                current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if in_attempt and current_frame_num % (FRAME_SKIP * 50) == 0:
                    safe_print(f"  [{os.path.basename(video_file)}] frame {current_frame_num}/{total_frames}, {current_sec:.1f}s, in attempt #{attempt_number}")
                elif current_frame_num % (FRAME_SKIP * 100) == 0:
                    safe_print(f"  [{os.path.basename(video_file)}] frame {current_frame_num}/{total_frames}, {current_sec:.1f}s")
                # Сохраняем оригинальный кадр для превью
                original_frame = frame.copy()
                
                # Установка области интереса
                if roi:
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Проверяем формат ROI
                    if len(roi) == 4:  # Прямоугольник (две точки: x1,y1,x2,y2)
                        left, top, right, bottom = roi
                        x1_px = int(frame_width * left / 100)
                        y1_px = int(frame_height * top / 100)
                        x2_px = int(frame_width * right / 100)
                        y2_px = int(frame_height * bottom / 100)
                        frame = frame[y1_px:y2_px, x1_px:x2_px]
                        # Также обрезаем оригинальный кадр для превью
                        original_frame = original_frame[y1_px:y2_px, x1_px:x2_px]
                    elif len(roi) > 4:  # Многоугольник (новый формат)
                        # Конвертируем точки из процентов в пиксели
                        polygon_points = [(int(frame_width * x / 100), int(frame_height * y / 100)) for x, y in roi]
                        
                        # Создаем маску многоугольника
                        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                        polygon_array = np.array(polygon_points, dtype=np.int32)
                        cv2.fillPoly(mask, [polygon_array], 255)
                        
                        # Применяем маску к кадру для детекции
                        frame = cv2.bitwise_and(frame, frame, mask=mask)
                        # Оригинальный кадр оставляем без маски для превью

                # Детекция с помощью YOLO
                # scale_factor = 0.3
                if frame.shape[1] > YOLO_MAX_WIDTH:  # Check if the current frame's width is greater than max
                    scale_factor = YOLO_MAX_WIDTH / frame.shape[1]
                    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                    small_frame_original = cv2.resize(original_frame, None, fx=scale_factor, fy=scale_factor)
                else:
                    small_frame = frame.copy()  # No resizing needed
                    small_frame_original = original_frame.copy()

                results = model(small_frame, verbose=False)

                # Проверка наличия объектов
                detections = []
                for r in results:
                    boxes = r.boxes
                    if best_frame_confidence == 0:
                        best_frame = small_frame_original  # Используем оригинальный кадр для превью
                        person_frame = small_frame_original
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls]
                        if class_name == 'person' and best_frame_confidence <= conf and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            best_frame_confidence = conf
                            best_frame = small_frame_original  # Используем оригинальный кадр для превью
                            person_frame = small_frame_original[y1:y2, x1:x2]
                            person_bbox = box.xyxy[0].cpu().numpy()
                        if class_name in YOLO_TARGET_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            detections.append({'class': class_name})

                has_detections = best_frame_confidence > 0 and len(detections) > 0

                # запоминаем базовый фрейм без человека
                if not in_attempt:
                    base_frame = small_frame_original  # Используем оригинальный кадр

                if has_detections:
                    frames_in_attempt += 1
                    frames_out_of_attempt = 0
                    if not in_attempt:
                        detection_start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if detection_start_time - (end_time or 0) >= MIN_PAUSE_DURATION:
                            in_attempt = True
                            start_time = detection_start_time
                else:
                    frames_out_of_attempt += 1
                    if in_attempt and frames_out_of_attempt >= MIN_PAUSE_DURATION * fps / FRAME_SKIP:
                        in_attempt = False
                        best_frame_confidence = 0
                        end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                        result, attempt_number = _finalize_attempt(
                            attempt_number, start_time, end_time, total_frames, fps,
                            person_bbox, base_frame, best_frame, person_frame,
                            small_frame_original, video_file, callback)
                        if result is not None:
                            best_frame = None
                            base_frame = None
                        if result is False:
                            cap.release()
                            return

        if in_attempt and start_time is not None:
            end_time = total_frames / fps
            safe_print(f"  [{os.path.basename(video_file)}] Финализация попытки #{attempt_number} на конце видео ({end_time:.1f}s)")
            result, attempt_number = _finalize_attempt(
                attempt_number, start_time, end_time, total_frames, fps,
                person_bbox, base_frame, best_frame, person_frame,
                small_frame_original, video_file, callback)

        cap.release()
