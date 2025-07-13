import colorsys
import platform
import os
import sys
import subprocess
import cv2
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.colors as mcolors
import re

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
    css4_colors = mcolors.CSS4_COLORS
    names = list(css4_colors.keys())
    rgb_values = [mcolors.hex2color(css4_colors[name]) for name in names]
    kdtree = KDTree(rgb_values)
    distance, index = kdtree.query(requested_color)
    return names[index], distance
