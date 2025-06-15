import platform
import os

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
