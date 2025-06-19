import platform
import os
import sys

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