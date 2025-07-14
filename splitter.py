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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_limit):
                break

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP == 0:
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

                        if end_time - start_time >= MIN_ATTEMPT_DURATION:
                            start1 = max(0, start_time - ATTEMPT_START_PADDING)
                            end1 =  min(end_time + ATTEMPT_END_PADDING, (total_frames * fps)/1000.0)


                            x1, y1, x2, y2 = map(int, person_bbox)
                            try:
                                bf = base_frame[y1:y2, x1:x2]
                            except:
                                bf = small_frame_original[y1:y2, x1:x2]

                            end1

                            attempt = AttemptInfo(number = attempt_number,
                                                  base_frame=bf,
                                                  best_frame=best_frame,
                                                  person_frame=person_frame,
                                                  person_bbox=person_bbox,
                                                  source_video=video_file,
                                                  start = start1, end=end1)
                            # output_file = os.path.join(output_folder, f"{attempt.number:04d}.mp4")
                            # safe_print(f"Saving attempt {attempt_number} from {attempt.start:.2f}s to {attempt.end:.2f}s (duration: {attempt.duration():.2f}s)")
                            # subprocess.run([get_ffmpeg_path(), "-i", video_file, "-ss", str(attempt.start), "-to", str(attempt.end), "-c", "copy", output_file], check=True)
                            # safe_print(f"Saved attempt {attempt.number} from {attempt.start:.2f}s to {attempt.end:.2f}s (duration: {attempt.duration():.2f}s)")
                            attempt_number += 1
                            need_stop = not callback(attempt)
                            # need_stop = callback(output_file)
                            best_frame = None
                            base_frame = None
                            if need_stop:
                                cap.release()
                                return

        cap.release()
