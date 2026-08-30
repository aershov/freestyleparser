import cv2
import os
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import glob
import subprocess
import threading
from queue import Queue

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Порог уверенности для детекции объектов
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_FRAME_SCALE_FACTOR = 0.2  # Масштаб для уменьшения кадра перед детекцией

# Конфигурация области интереса (ROI)
ROI_PERCENTAGES = (0, 23, 95, 90)  # (left, top, right, bottom) в процентах от размера кадра

# Конфигурация обработки видео
FRAME_SKIP = 25  # Обрабатываем каждый N-й кадр
MIN_DETECTION_TIME = 1  # Минимальное время (в секундах) с детекциями для начала попытки
MIN_ATTEMPT_DURATION = 3  # Минимальная длительность попытки в секундах
MIN_PAUSE_DURATION = 3  # Минимальная пауза между попытками в секундах

ATTEMPT_START_PADDING = 2  # Запас времени (в секундах) к началу попытки
ATTEMPT_END_PADDING = 1 # Запас времени (в секундах) от конца попытки

def safe_print(*args, **kwargs):
    """Потокобезопасный вывод в консоль"""
    print(*args, **kwargs)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video files to extract athlete attempts')
    parser.add_argument('--input', '-i', type=str, default='.',
                        help='Input directory or file path (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit processing to first N seconds of all videos (default: process all)')
    parser.add_argument('--inputFiles', '-f', type=str, default=None,
                        help='Filter input files by mask or specific files (semicolon-separated). Example: "*.MTS" or "file1.MTS;file2.MTS"')
    return parser.parse_args()

def get_video_files(input_path, file_filter=None):
    """Получает список видео файлов для обработки"""
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    elif path.is_dir():
        if file_filter:
            if ';' in file_filter:
                # Список конкретных файлов
                files = []
                for file_name in file_filter.split(';'):
                    file_path = path / file_name.strip()
                    if file_path.exists():
                        files.append(str(file_path))
            else:
                # Маска файлов
                files = glob.glob(str(path / file_filter))
        else:
            # Все .MTS файлы
            files = glob.glob(str(path / "*.MTS"))
        
        files.sort()  # Сортировка по имени файла
        safe_print("Files will be processed in the following order:")
        for file in files:
            safe_print(f"  {Path(file).name}")
        return files
    return []

def detect_objects_yolo(frame, model, roi_percentages):
    frame_height, frame_width = frame.shape[:2]
    x1_px = int(frame_width * roi_percentages[0] / 100)
    y1_px = int(frame_height * roi_percentages[1] / 100)
    x2_px = int(frame_width * roi_percentages[2] / 100)
    y2_px = int(frame_height * roi_percentages[3] / 100)
    roi = frame[y1_px:y2_px, x1_px:x2_px]
    small_frame = cv2.resize(roi, None, fx=YOLO_FRAME_SCALE_FACTOR, fy=YOLO_FRAME_SCALE_FACTOR)
    results = model(small_frame, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            x1, y1, x2, y2 = [int(coord / YOLO_FRAME_SCALE_FACTOR) for coord in [x1, y1, x2, y2]]
            x1 += x1_px
            x2 += x1_px
            y1 += y1_px
            y2 += y1_px
            if class_name in YOLO_TARGET_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name
                })
    return detections

# Очередь для создания файлов попыток
attempt_queue = Queue()

def save_attempt_worker():
    """Рабочий поток для создания файлов попыток"""
    attempt_count = 0
    while True:
        attempt_data = attempt_queue.get()
        if attempt_data is None:
            break
        video_path, start, end, output_dir = attempt_data
        attempt_count += 1
        safe_print(f"Saving attempt {attempt_count} from {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
        output_path = os.path.join(output_dir, f"{attempt_count:04d}.mp4")
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start),
            '-to', str(end),
            '-c', 'copy',
            output_path
        ]
        subprocess.run(cmd)
        safe_print(f"Saved attempt {attempt_count} from {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")

def process_video(video_path, output_dir, model, time_limit=None, total_duration=0):
    """Обрабатывает видео и вырезает попытки с помощью ffmpeg"""
    safe_print(f"Processing video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        safe_print(f"Error: Could not open video file {video_path}")
        return total_duration

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Выводим информацию о видео
    safe_print(f"\nVideo information:")
    safe_print(f"Resolution: {frame_width}x{frame_height}")
    safe_print(f"FPS: {fps:.2f}")
    safe_print(f"Total frames: {frame_count}")
    safe_print(f"Duration: {duration:.2f} seconds")
    if time_limit is not None:
        safe_print(f"Processing limited to first {time_limit} seconds")
        duration = min(duration, time_limit)

    # Список попыток
    attempt_start = None
    no_detection_count = 0
    detection_count = 0
    frame_index = 0
    last_attempt_end = 0  # Время окончания последней попытки

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if time_limit is not None and total_duration + current_time > time_limit:
            break

        if frame_index % FRAME_SKIP == 0:  # Обрабатываем каждый N-й кадр
            detections = detect_objects_yolo(frame, model, ROI_PERCENTAGES)
            has_detections = any(d['class'] in YOLO_TARGET_CLASSES for d in detections)
            
            if has_detections:
                # Проверяем, прошло ли достаточно времени с последней попытки
                if current_time - last_attempt_end >= MIN_PAUSE_DURATION:
                    detection_count += 1
                    if detection_count >= MIN_DETECTION_TIME * fps/FRAME_SKIP:  # Минимальное время с детекциями
                        if attempt_start is None:
                            attempt_start = current_time
                        no_detection_count = 0
            else:
                no_detection_count += 1
                if no_detection_count >= MIN_PAUSE_DURATION * fps/FRAME_SKIP:  # Минимальная пауза между попытками
                    if attempt_start is not None:
                        # Проверяем минимальную длительность попытки
                        if current_time - attempt_start >= MIN_ATTEMPT_DURATION:
                            start1 = max(0, attempt_start - ATTEMPT_START_PADDING)
                            end1 = current_time - ATTEMPT_END_PADDING
                            attempt_queue.put((video_path, start1, end1, output_dir))
                            last_attempt_end = current_time
                        attempt_start = None
                    no_detection_count = 0
                    detection_count = 0

        frame_index += 1

    cap.release()
    return total_duration + duration

def main():
    args = parse_arguments()
    model = YOLO('models/yolo11s-seg.pt')
    video_files = get_video_files(args.input, args.inputFiles)
    if not video_files:
        safe_print(f"No video files found in {args.input}")
        return

    # Создаем выходную директорию, если она не существует
    os.makedirs(args.output, exist_ok=True)

    # Запускаем рабочий поток для создания файлов попыток
    save_thread = threading.Thread(target=save_attempt_worker)
    save_thread.start()

    total_duration = 0
    for video_file in video_files:
        safe_print(f"Processing {video_file}...")
        total_duration = process_video(video_file, args.output, model, args.limit, total_duration)

    # Сигнализируем рабочему потоку о завершении
    attempt_queue.put(None)
    save_thread.join()

if __name__ == "__main__":
    main() 