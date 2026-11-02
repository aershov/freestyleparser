import colorsys
import platform
import os
import sys
import json
import subprocess
import shutil
import tempfile
import cv2
import numpy as np
from collections import defaultdict
import re

# Дефолты выверены на съёмке 2026-11-01 (Открытие сезона): ловят короткие
# трюки и не дробят серию входов при рыхлой детекции. Живут здесь (без
# тяжёлых импортов torch/ultralytics), splitter.py переэкспортирует.
DEFAULT_PROCESSING_PARAMS = {
    'min_attempt_duration': 2.0,   # Минимальная длительность попытки в секундах
    'min_pause_duration': 2.0,     # Минимальная пауза между попытками в секундах
    'attempt_start_padding': 1.5,  # Запас времени (в секундах) к началу попытки
    'attempt_end_padding': 1.5,    # Запас времени (в секундах) от конца попытки
    'min_detection_strength': 0.3, # Минимальная сила сигнала (0-1) для начала/продолжения попытки
    'scan_interval': 1.0,          # Шаг сканирования (ручной режим), сек реального времени
    'scan_threshold': 0.25,        # Порог YOLO при сканировании (ниже порога обычного анализа)
    'reid_embed_rate': 3.0,        # OSNet-эмбеддинги внутри попытки, шт/сек реала (5 = каждый сэмпл)
    'precise_left_margin': 3.5,    # Насколько влево от кандидата искать точное начало, сек
}

CAMERA_PATHS = [
    {
        "name": "Sony AVCHD Cam",
        "path": "PRIVATE/AVCHD/BDMV/STREAM/"
    },
    {
        "name": "GoPro HERO",
        "path": "DCIM/100GOPRO"
    },
    {
        "name": "DJI Osmo Action",
        "path": "DJI/MP4"
    },
    {
        "name": "DJI Osmo Action 1",
        "path": "DICM/100MEDIA"
    },
    {
        "name": "Canon Camera",
        "path": "DCIM/100CANON"
    },
    {
        "name": "Other",
        "path": "DCIM"
    }

]

def get_ffmpeg_path():
    """Получает путь к ffmpeg из системы или из папки bin"""
    # Сначала проверяем системные пути
    if platform.system() == 'Windows':
        system_paths = os.environ.get('PATH', '').split(';')
    else:
        system_paths = os.environ.get('PATH', '').split(':')

    # Проверяем системные пути
    for path in system_paths:
        ffmpeg_path = os.path.join(path, 'ffmpeg')
        if platform.system() == 'Windows':
            ffmpeg_path += '.exe'
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path

    # Если не нашли в системе, проверяем папку bin
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg')
    if platform.system() == 'Windows':
        bin_path += '.exe'
    if os.path.exists(bin_path):
        return bin_path

    return None

def get_ffprobe_path():
    """Получает путь к ffprobe из системы или из папки bin"""
    # Сначала проверяем системные пути
    if platform.system() == 'Windows':
        system_paths = os.environ.get('PATH', '').split(';')
    else:
        system_paths = os.environ.get('PATH', '').split(':')

    # Проверяем системные пути
    for path in system_paths:
        ffprobe_path = os.path.join(path, 'ffprobe')
        if platform.system() == 'Windows':
            ffprobe_path += '.exe'
        if os.path.exists(ffprobe_path):
            return ffprobe_path

    # Если не нашли в системе, проверяем папку bin
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffprobe')
    if platform.system() == 'Windows':
        bin_path += '.exe'
    if os.path.exists(bin_path):
        return bin_path

    return None


def _parse_rate(value):
    """Разбирает строку вида '96/1' или '24.0' в float"""
    try:
        num, _, den = str(value).partition('/')
        den = float(den) if den else 1.0
        return float(num) / den if den else 0.0
    except Exception:
        return 0.0


def _ffprobe_stream_info(video_path, entries):
    """Возвращает json-инфо о первом видео-потоке через ffprobe или None"""
    ffprobe = get_ffprobe_path()
    if not ffprobe or not video_path or not os.path.exists(video_path):
        return None
    try:
        cmd = [ffprobe, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', f'stream={entries}',
               '-of', 'json', video_path]
        out = subprocess.check_output(cmd, timeout=30).decode()
        streams = (json.loads(out) or {}).get('streams') or []
        return streams[0] if streams else None
    except Exception:
        return None


def detect_slowmo_factor(video_path):
    """Пытается определить по метаданным, является ли видео слоумо-записью.

    Признак - VFR-файл: r_frame_rate существенно выше avg_frame_rate
    (например, 96 к/с при 24 к/с воспроизведения). Возвращает целочисленный
    коэффициент замедления (1 - обычное видео). Для файлов, где слоумо
    зашито только в таймстампах контейнера (Panasonic GH4 VFR), метаданные
    неотличимы от обычного 24 к/с - тогда вернётся 1 и нужен ручной чекбокс.

    Interlaced-источники (1080/50i: 50 полей = 25 кадров) дают такой же
    профиль r/avg = 2, но это НЕ слоумо - для них всегда вернём 1
    (настоящее VFR-слоумо пишется в progressive).
    """
    entries = 'r_frame_rate,avg_frame_rate,field_order'
    info = _ffprobe_stream_info(video_path, entries)
    if not info:
        return 1
    if str(info.get('field_order') or '').lower() in ('tt', 'bb', 'tb', 'bt'):
        return 1
    r_rate = _parse_rate(info.get('r_frame_rate'))
    avg_rate = _parse_rate(info.get('avg_frame_rate'))
    if avg_rate > 1 and r_rate / avg_rate >= 1.5:
        return max(1, int(round(r_rate / avg_rate)))
    return 1


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite, работающий с не-ASCII путями на Windows (кириллица в
    имени пользователя ломает fopen внутри OpenCV - imwrite молча не пишет).
    Кодируем кадр в память и пишем файлом Python, который использует
    юникодные API. Возвращает True при успехе."""
    try:
        ext = os.path.splitext(path)[1] or '.jpg'
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def open_capture(video_path, backend=None):
    """cv2.VideoCapture с обходом проблемы Windows: если путь содержит
    не-ASCII (кириллица в имени пользователя) и открыть не удалось -
    копируем исходник во временную папку с ASCII-именем и открываем оттуда."""
    cap = cv2.VideoCapture(video_path, backend) if backend is not None \
        else cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap
    cap.release()
    if platform.system() != 'Windows':
        return cap
    try:
        video_path.encode('ascii')
        return cap  # путь ASCII - проблема не в нём
    except UnicodeEncodeError:
        pass
    try:
        import hashlib
        tmp_dir = os.path.join(tempfile.gettempdir(), 'FreestyleParserCaps')
        os.makedirs(tmp_dir, exist_ok=True)
        digest = hashlib.md5(video_path.encode('utf-8')).hexdigest()[:12]
        tmp_path = os.path.join(tmp_dir, digest + os.path.splitext(video_path)[1])
        if not os.path.exists(tmp_path) or \
                os.path.getsize(tmp_path) != os.path.getsize(video_path):
            shutil.copy2(video_path, tmp_path)
        return cv2.VideoCapture(tmp_path, backend) if backend is not None \
            else cv2.VideoCapture(tmp_path)
    except Exception:
        return cap


def open_capture_seekable(video_path):
    """VideoCapture, в котором работает произвольный seek. Некоторые контейнеры
    (AVCHD .mts) OpenCV-FFMPEG не умеет искать по кадрам/времени (read у 98%
    длительности возвращает False) - плеер для последних попыток показывал
    пустой экран. Пробуем дефолтный бэкенд, при неудаче пробного seek -
    перебираем платформенные (на macOS AVFOUNDATION ищет точно)."""
    cap = open_capture(video_path)
    if cap.isOpened() and _seek_probe_ok(cap):
        return cap
    try:
        cap.release()
    except Exception:
        pass
    backends = []
    if platform.system() == 'Darwin':
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_FFMPEG]
    elif platform.system() == 'Windows':
        backends = [cv2.CAP_MSMF, cv2.CAP_FFMPEG, cv2.CAP_DSHOW]
    else:
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
    for be in backends:
        try:
            cap = cv2.VideoCapture(video_path, be)
        except Exception:
            continue
        if cap.isOpened() and _seek_probe_ok(cap):
            return cap
        try:
            cap.release()
        except Exception:
            pass
    return open_capture(video_path)


def _seek_probe_ok(cap):
    """Пробный seek на ~90% длительности: read должен вернуть кадр."""
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not fps or not n or n <= 10:
            return True  # нечего проверять - короткий файл/нет данных
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.9))
        ok, _f = cap.read()
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return bool(ok) and pos > (n * 0.9 / fps) * 0.5
    except Exception:
        return True


def has_audio_stream(video_path):
    """Проверяет наличие аудио-дорожки в видео через ffprobe"""
    ffprobe = get_ffprobe_path()
    if not ffprobe or not video_path or not os.path.exists(video_path):
        return False
    try:
        cmd = [ffprobe, '-v', 'error', '-select_streams', 'a',
               '-show_entries', 'stream=index',
               '-of', 'csv=p=0', video_path]
        return bool(subprocess.check_output(cmd, timeout=30).decode().strip())
    except Exception:
        return False


def find_default_video_folder():
    """Ищет подключённые диски и флешки с видеофайлами камер"""
    candidates = []

    if sys.platform == "darwin":
        # Ищем все подключённые диски
        mount_points = ["/Volumes/" + f for f in os.listdir("/Volumes/") if f not in (".", "..")]

        for base in mount_points:
            for cam in CAMERA_PATHS:
                test_path = os.path.join(base, cam["path"])
                if os.path.exists(test_path):
                    candidates.append(test_path)

    elif sys.platform == "win32":
        # Ищем все диски на Windows
        import string
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        for base in drives:
            for cam in CAMERA_PATHS:
                test_path = os.path.join(base, cam["path"])
                if os.path.exists(test_path):
                    candidates.append(test_path)

    else:
        # Linux / другие ОС — можно добавить путь домашней папки
        home = os.path.expanduser("~")
        for cam in CAMERA_PATHS:
            test_path = os.path.join(home, cam["path"])
            if os.path.exists(test_path):
                candidates.append(test_path)

    # Возвращаем первый найденный подходящий путь
    if candidates:
        return candidates[0]

    # Если ничего не нашли — текущая папка
    return os.path.expanduser("~")

def get_next_attempt_number(folder):
    if not os.path.exists(folder):
        return 1
    pattern = re.compile(r'^(\d{4})\.mp4$')  # ищем файлы формата "0001.mp4"
    max_number = 0
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            try:
                num = int(match.group(1))  # например, "0001" → 1
                if max_number < num:
                    max_number = num
            except ValueError:
                continue
    return max_number + 1

def get_dominant_colors(frame, k=5, threshold=10):
    """
    Возвращает список доминирующих цветов в формате [(имя_цвета, процент), ...]
    :param frame: кадр OpenCV (BGR)
    :param k: количество кластеров
    :param threshold: минимальный % площади для сохранения цвета
    """

    # --- Подготовка кадра ---
    resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    hsv_flat = hsv.reshape((h * w, 3))

    # --- Кластеризация ---
    from sklearn.cluster import KMeans  # ленивый: sklearn тянется 3-4с
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(hsv_flat)
    cluster_centers = kmeans.cluster_centers_.astype(int)

    values, counts = np.unique(labels, return_counts=True)
    total_pixels = h * w

    color_groups = {}

    for label in values:
        count = counts[label]
        if count / total_pixels < threshold / 100:
            continue  # пропускаем малые области

        h_val, s_val, v_val = cluster_centers[label]

        # --- Игнорируем фоновые цвета ---
        if s_val < 40 and v_val > 200:
            continue  # белый
        elif s_val < 30 and v_val < 30:
            continue  # черный
        elif s_val < 30 and v_val < 200:
            continue  # серый фон

        # --- Определяем имя цвета ---
        name = hsv_to_name(h_val, s_val, v_val)

        # --- Группируем по имени цвета ---
        if name not in color_groups:
            color_groups[name] = {
                "count": count,
                "color_hsv": (h_val, s_val, v_val),
                "examples": [(h_val, s_val, v_val)]
            }
        else:
            color_groups[name]["count"] += count
            color_groups[name]["examples"].append((h_val, s_val, v_val))

    # --- Считаем общий процент для каждого цвета ---
    final_colors = []
    for name, data in color_groups.items():
        percent = data["count"] / total_pixels * 100
        if percent >= threshold:
            final_colors.append((name, percent))

    return sorted(final_colors, key=lambda x: -x[1])


def hsv_to_name0(h, s, v):
    """
    Конвертирует HSV в имя цвета, с учётом оттенка, насыщенности и яркости.
    Теперь умеет различать светло-зеленый и темно-зеленый
    """

    # Ахроматический цвет (серый)
    if s < 20:
        return "серый"

    # Яркость очень низкая → чёрный
    if v < 30:
        return "черный"

    # Яркость высокая, но нет насыщенности → белый
    if s < 30 and v > 180:
        return "белый"

    # === Цвета по Hue (H) ===
    if h < 5 or h > 170:
        return "красный"
    elif h < 15:
        return "оранжевый"
    elif h < 35:
        return "желтый"
    elif h < 85:
        # Зеленый диапазон: 35–85
        if v > 150:
            return "светло-зеленый"
        else:
            return "темно-зеленый"
    elif h < 130:
        # Синий диапазон: 85–130
        if s > 100:
            return "синий"
        else:
            if  v < 50:
                return "черный"
            return "голубой"
    elif h < 150:
        return "фиолетовый"
    else:
        return "розовый"

def extract_athlete_difference(frame_bg, frame_with, bbox=None):
    """
    Находит атлета по разнице между кадром без человека и с человеком.
    Если дан bbox — использует его как ROI.
    :param frame_bg: фоновый кадр (без атлета)
    :param frame_with: кадр с атлетом
    :param bbox: [x1, y1, x2, y2] — если есть YOLO-детекция
    :return: ROI области атлета
    """

    # --- Проверка совместимости кадров ---
    if frame_bg is None or frame_with is None:
        print("[ERROR] Один из кадров не загружен.")
        return None

    if frame_bg.shape != frame_with.shape:
        print(f"[WARNING] Кадры разного размера. Выравниваем...")
        frame_bg = cv2.resize(frame_bg, (frame_with.shape[1], frame_with.shape[0]))

    # --- Если есть BBox, создаём маску только по нему ---
    mask = np.zeros((frame_with.shape[0], frame_with.shape[1]), dtype=np.uint8)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_with.shape[1], x2)
        y2 = min(frame_with.shape[0], y2)
        mask[y1:y2, x1:x2] = 255  # маска только в зоне bbox
    else:
        # --- Разница между кадрами ---
        diff = cv2.absdiff(frame_bg, frame_with)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

        # --- Морфологические операции для очистки шума ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # --- Бинаризуем маску, чтобы получить только изменение ---
        coords = cv2.findNonZero(binary_mask)
        if coords is None:
            print("[INFO] Не найдено изменений между кадрами.")
            return None

        x, y, w, h = cv2.boundingRect(coords)
        mask[y:y+h, x:x+w] = 255  # выделяем только изменённую область

    # --- Применяем маску к кадру с атлетом ---
    athlete_roi = cv2.bitwise_and(frame_with, frame_with, mask=mask)

    return athlete_roi


def extract_athlete_difference0(frame_bg, frame_with, bbox=None):
    """
    Находит разницу между кадром без атлета и с атлетом.
    Если дан bbox — использует его как маску.
    Возвращает ROI только с атлетом и лодкой
    """
    # --- Шаг 1: Предобработка ---
    # if frame_bg.shape != frame_with.shape:
    #     frame_bg = cv2.resize(frame_bg, (frame_with.shape[1], frame_with.shape[0]))

    # --- Шаг 2: Разница между кадрами ---
    diff = cv2.absdiff(frame_bg, frame_with)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # --- Шаг 3: Используем bbox, если он передан ---
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        person_mask = np.zeros_like(mask)
        person_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, mask, mask=person_mask)

    # --- Шаг 4: Применяем маску к кадру ---
    athlete_roi = cv2.bitwise_and(frame_with, frame_with, mask=mask)

    # --- Шаг 5: Вырезаем только область с разницей ---
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("[INFO] Не найдено изменений между кадрами.")
        return []

    x, y, w, h = cv2.boundingRect(coords)
    cropped = frame_with[y:y+h, x:x+w]

    return cropped

def hsv_to_name(h, s, v):
    """Преобразует HSV в имя цвета с точностью"""

    # --- Ахроматические цвета ---
    if s < 30:
        if v > 230:
            return "белый"
        elif v < 30:
            return "черный"
        else:
            return "серый"

    # --- Яркость низкая → темные цвета ---
    if v < 80:
        if h < 15 or h > 165:
            return "темно-красный"
        elif h < 35:
            return "темно-оранжевый"
        elif h < 85:
            return "темно-зеленый"
        elif h < 130:
            return "темно-синий"
        elif h < 150:
            return "фиолетовый"
        else:
            return "розовый"

    # --- Средняя яркость ---
    elif v < 200:
        if h < 5 or h > 170:
            return "красный"
        elif h < 25:
            return "оранжевый"
        elif h < 35:
            return "желтый"
        elif h < 85:
            return "зеленый"
        elif h < 130:
            return "синий"
        elif h < 150:
            return "фиолетовый"
        else:
            return "розовый"

    # --- Высокая яркость → светлые тона ---
    else:
        if h < 5 or h > 170:
            return "розово-красный"
        elif h < 25:
            return "оранжевый"
        elif h < 35:
            return "желтый"
        elif h < 85:
            return "светло-зеленый"
        elif h < 130:
            return "светло-синий"
        elif h < 150:
            return "светло-фиолетовый"
        else:
            return "светло-розовый"


def hsv_to_name1(h,s,v):
    rgb = hsv_to_rgb(h,s,v)
    return rgb_to_name(rgb)

def hsv_to_rgb(h,s,v):
    rgb = tuple(
        round(i * 255)
        for i in colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    )
    return rgb


def rgb_to_name(rgb):
    rgb_normalized = tuple([x / 255 for x in rgb])
    color_name, distance = closest_color_name(rgb_normalized)
    if not distance == 0:
        color_name = color_name + "(approx)"
    return color_name

def closest_color_name(requested_color):
    from scipy.spatial import KDTree        # ленивые импорты тяжёлых пакетов
    import matplotlib.colors as mcolors
    css4_colors = mcolors.CSS4_COLORS
    names = list(css4_colors.keys())
    rgb_values = [mcolors.hex2color(css4_colors[name]) for name in names]
    kdtree = KDTree(rgb_values)
    distance, index = kdtree.query(requested_color)
    return names[index], distance
