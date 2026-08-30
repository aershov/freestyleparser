import cv2
import os
import re
import subprocess
import sys
from ultralytics import YOLO
from utils import get_ffmpeg_path

# Конфигурация YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Порог уверенности для детекции объектов
YOLO_TARGET_CLASSES = ['person', 'boat', 'surfboard']  # Классы объектов для детекции
YOLO_MAX_WIDTH = 640  # уменьшаем кадр перед анализом в YOLO до стольких пикселей
# 640 = нативный вход yolov8; при 200px атлет в пене становится неразличим (проверено на GoPro-видео)


# Конфигурация обработки видео
FRAME_SKIP = 25  # Анализируем каждый N-й кадр //TODO надо поменять на указания N кадров в секунду анализа, чтобы не зависеть от частоты кадров в видео
MIN_ATTEMPT_DURATION = 3  # Минимальная длительность попытки в секундах
MIN_PAUSE_DURATION = 3  # Минимальная пауза между попытками в секундах (если потеряли атлета меньше чем на столько сек, то попытка продолжается, если больше - то новая попытка)

ATTEMPT_START_PADDING = 2  # Запас времени (в секундах) к началу попытки
ATTEMPT_END_PADDING = 0  # Запас времени (в секундах) от конца попытки

DEFAULT_FPS = 25.0  # Если fps не удалось прочитать из файла


def safe_print(*args, **kwargs):
    """Потокобезопасный вывод в консоль"""
    print(*args, **kwargs)


def get_next_attempt_number(output_folder):
    """Продолжаем нумерацию после уже существующих файлов, чтобы не перезаписывать их"""
    max_number = 0
    if os.path.isdir(output_folder):
        for name in os.listdir(output_folder):
            match = re.fullmatch(r'(\d+)\.mp4', name)
            if match:
                max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def cut_attempt(video_file, output_file, start, end):
    """Вырезает попытку с помощью ffmpeg (stream copy, быстрый seek до входа)"""
    subprocess.run([
        get_ffmpeg_path(),
        '-ss', f'{start:.3f}',  # до -i: быстрый seek по ключевым кадрам вместо чтения файла с начала
        '-i', video_file,
        '-t', f'{max(end - start, 0.1):.3f}',  # длительность, а не абсолютная метка: после seek метки сбрасываются
        '-c', 'copy',
        '-y',  # без вопроса о перезаписи - иначе ffmpeg зависнет в ожидании ответа
        output_file,
    ], check=True)


def save_attempt(video_file, output_folder, number, start_time, end_time, callback):
    """Сохраняет попытку; возвращает False, если обработку попросили остановить"""
    start = max(0, start_time - ATTEMPT_START_PADDING)
    end = end_time + ATTEMPT_END_PADDING
    output_file = os.path.join(output_folder, f'{number:04d}.mp4')
    safe_print(f"Saving attempt {number} from {start:.2f}s to {end:.2f}s (duration: {end - start:.2f}s)")
    cut_attempt(video_file, output_file, start, end)
    safe_print(f"Saved attempt {number}")
    return callback(output_file)


def crop_roi(frame, roi):
    """Вырезает из кадра область интереса (в процентах от размера кадра)"""
    left, top, right, bottom = roi
    frame_height, frame_width = frame.shape[:2]
    x1 = int(frame_width * left / 100)
    y1 = int(frame_height * top / 100)
    x2 = int(frame_width * right / 100)
    y2 = int(frame_height * bottom / 100)
    return frame[y1:y2, x1:x2]


def detect_has_target(model, frame):
    """Проверяет, есть ли в кадре целевой объект (человек/лодка/доска)"""
    if frame.shape[1] > YOLO_MAX_WIDTH:
        scale_factor = YOLO_MAX_WIDTH / frame.shape[1]
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    else:
        small_frame = frame
    results = model(small_frame, verbose=False)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]
            if class_name in YOLO_TARGET_CLASSES and conf >= YOLO_CONFIDENCE_THRESHOLD:
                return True
    return False


def open_capture(video_file):
    """Открывает видео. На Windows бэкенд FFMPEG часто не читает файлы камер
    (например, GoPro: чтение обрывается на первых кадрах с "packet read max
    attempts exceeded"), поэтому сначала пробуем Media Foundation."""
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(video_file, cv2.CAP_MSMF)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(video_file)


def process_video(input_paths, output_folder, roi, callback,
                  progress_callback=None, should_continue=None):
    """Обрабатывает видео и вырезает попытки.

    callback(output_file) вызывается после сохранения каждой попытки;
    если вернёт False - обработка останавливается.
    progress_callback(file_index, file_count, frame_index, total_frames) - прогресс обработки.
    should_continue() - если вернёт False, обработка останавливается (кнопка "Остановить").
    """
    if not get_ffmpeg_path():
        raise RuntimeError("FFmpeg не найден в системе или в папке bin")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError("'yolov8n.pt' не найден")
    model = YOLO(model_path)

    # sorted() не мутирует список вызывающего (self.selected_files)
    video_files = sorted(input_paths, key=os.path.basename)

    os.makedirs(output_folder, exist_ok=True)
    attempt_number = get_next_attempt_number(output_folder)

    for file_index, video_file in enumerate(video_files):
        safe_print(f"Processing {video_file}...")
        cap = open_capture(video_file)
        if not cap.isOpened():
            safe_print(f"Не удалось открыть файл: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pause_frames_limit = MIN_PAUSE_DURATION * fps / FRAME_SKIP

        in_attempt = False
        start_time = None
        last_detection_time = None
        frames_out_of_attempt = 0

        frame_index = -1
        while True:
            if should_continue is not None and not should_continue():
                cap.release()
                return False

            # grab() не декодирует кадр - намного быстрее read() для пропускаемых кадров
            if not cap.grab():
                break
            frame_index += 1

            if frame_index % FRAME_SKIP != 0:
                continue

            if progress_callback is not None:
                progress_callback(file_index, len(video_files), frame_index, total_frames)

            ret, frame = cap.retrieve()
            if not ret:
                break
            # Время считаем по счётчику кадров: CAP_PROP_POS_MSEC ненадёжен у некоторых бэкендов
            current_time = frame_index / fps

            if roi:
                frame = crop_roi(frame, roi)
                if frame.size == 0:
                    continue  # вырожденная область интереса

            has_detections = detect_has_target(model, frame)

            if has_detections:
                frames_out_of_attempt = 0
                last_detection_time = current_time
                if not in_attempt:
                    in_attempt = True
                    start_time = current_time
            elif in_attempt:
                frames_out_of_attempt += 1
                if frames_out_of_attempt >= pause_frames_limit:
                    # Атлет пропал дольше MIN_PAUSE_DURATION - попытка закончилась
                    # Концом считаем момент последней детекции, а не момент превышения паузы
                    in_attempt = False
                    frames_out_of_attempt = 0
                    if last_detection_time - start_time >= MIN_ATTEMPT_DURATION:
                        if not save_attempt(video_file, output_folder, attempt_number,
                                            start_time, last_detection_time, callback):
                            cap.release()
                            return False
                        attempt_number += 1

        # Файл кончился, пока попытка ещё шла - не теряем её
        if (in_attempt and start_time is not None and last_detection_time is not None
                and last_detection_time - start_time >= MIN_ATTEMPT_DURATION):
            if not save_attempt(video_file, output_folder, attempt_number,
                                start_time, last_detection_time, callback):
                cap.release()
                return False
            attempt_number += 1

        if total_frames and frame_index < total_frames * 0.95:
            safe_print(f"ВНИМАНИЕ: прочитано {frame_index + 1} кадров из {total_frames} "
                       f"заявленных - файл, возможно, обработан не полностью")

        cap.release()

    return True
