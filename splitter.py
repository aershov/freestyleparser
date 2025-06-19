import cv2
import numpy as np
import os
import subprocess
from setuptools.errors import FileError
from ultralytics import YOLO
from utils import get_ffmpeg_path

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Порог уверенности для детекции объектов
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 200  # уменьшаем кадр перед анализом в YOLO до стольких пикселей


# Конфигурация обработки видео
FRAME_SKIP = 25  # Обрабатываем каждый N-й кадр //TODO надо поменять на указания N кадров в секунду анализа, чтобы не зависеть от частоты кадров в видео
MIN_DETECTION_TIME = 1  # Минимальное время (в секундах) с детекциями для начала попытки
MIN_ATTEMPT_DURATION = 3  # Минимальная длительность попытки в секундах
MIN_PAUSE_DURATION = 3  # Минимальная пауза между попытками в секундах (если потеряли атлета меньше чем на столько сек, то попытка продолжается, если больше - то новая попытка)

ATTEMPT_START_PADDING = 2  # Запас времени (в секундах) к началу попытки
ATTEMPT_END_PADDING = 0 # Запас времени (в секундах) от конца попытки

def safe_print(*args, **kwargs):
    """Потокобезопасный вывод в консоль"""
    print(*args, **kwargs)

def process_video(input_paths, output_folder, roi, callback):
    # Загрузка модели YOLO
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '', 'yolov8n.pt')
    if not os.path.exists(model_path):
        raise FileError("'yolov8n.pt' not found")
    model = YOLO(model_path)

    video_files = input_paths

    # Сортировка файлов по дате создания
    # video_files.sort(key=os.path.getctime)
    video_files.sort(key=os.path.basename, reverse=False)

    # Создание выходной папки, если она не существует
    os.makedirs(output_folder, exist_ok=True)

    attempt_number = 1
    for video_file in video_files:
        safe_print(f"Processing {video_file}...")
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_limit = int(limit * fps) if limit else total_frames
        frame_limit = total_frames

        in_attempt = False
        start_time = None
        end_time = None
        frames_in_attempt = 0
        frames_out_of_attempt = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (cap.get(cv2.CAP_PROP_POS_FRAMES) >= frame_limit):
                break

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP == 0:
                # Установка области интереса
                if roi:
                    # x1, y1, x2, y2 = roi
                    # frame = frame[y1:y2, x1:x2]
                    frame_height, frame_width = frame.shape[:2]
                    left, top, right, bottom = roi
                    x1_px = int(frame_width * left / 100)
                    y1_px = int(frame_height * top / 100)
                    x2_px = int(frame_width * right / 100)
                    y2_px = int(frame_height * bottom / 100)
                    frame = frame[y1_px:y2_px, x1_px:x2_px]

                # Детекция с помощью YOLO
                # scale_factor = 0.3
                if frame.shape[1] > YOLO_MAX_WIDTH:  # Check if the current frame's width is greater than 200
                    scale_factor = YOLO_MAX_WIDTH / frame.shape[1]
                    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                else:
                    small_frame = frame.copy()  # No resizing needed

                results = model(small_frame, verbose=False)

                # Проверка наличия объектов
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls]
                        if class_name in YOLO_TARGET_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            detections.append({'class': class_name})

                has_detections = len(detections) > 0

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
                        end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                        if end_time - start_time >= MIN_ATTEMPT_DURATION:
                            start1 = max(0, start_time - ATTEMPT_START_PADDING)
                            end1 = end_time + ATTEMPT_END_PADDING

                            # Вырезание попытки с помощью ffmpeg
                            output_file = os.path.join(output_folder, f"{attempt_number:04d}.mp4")
                            safe_print(f"Saving attempt {attempt_number} from {start1:.2f}s to {end1:.2f}s (duration: {end1-start1:.2f}s)")
                            subprocess.run([get_ffmpeg_path(), "-i", video_file, "-ss", str(start1), "-to", str(end1), "-c", "copy", output_file], check=True)
                            safe_print(f"Saved attempt {attempt_number} from {start1:.2f}s to {end1:.2f}s (duration: {end1-start1:.2f}s)")
                            attempt_number += 1
                            if not callback(output_file):
                                return

        cap.release()
