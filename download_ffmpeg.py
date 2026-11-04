import os
import platform
import requests
import zipfile
import tarfile
import shutil
import subprocess

def download_file(url, filename):
    """Скачивает файл по URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_ffmpeg():
    """Скачивает ffmpeg и ffprobe для текущей платформы"""
    system = platform.system()
    machine = platform.machine()
    
    # Создаем директорию для бинарных файлов
    bin_dir = 'bin'
    os.makedirs(bin_dir, exist_ok=True)
    
    if system == 'Darwin':
        # Статические сборки ffmpeg-static (пин за eugeneware/ffmpeg-static b6.1.1):
        # evermeet.cx умер (getrelease -> 410, версии -> 429 Too Many Connections)
        base = 'https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1'
        arch = 'arm64' if machine == 'arm64' else 'x64'

        download_file(f'{base}/ffmpeg-darwin-{arch}', os.path.join(bin_dir, 'ffmpeg'))
        download_file(f'{base}/ffprobe-darwin-{arch}', os.path.join(bin_dir, 'ffprobe'))
        os.chmod(os.path.join(bin_dir, 'ffmpeg'), 0o755)
        os.chmod(os.path.join(bin_dir, 'ffprobe'), 0o755)
        
    elif system == 'Windows':
        # Для Windows
        url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
        
        # Скачиваем архив
        zip_file = 'ffmpeg.zip'
        download_file(url, zip_file)
        
        # Распаковываем
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('temp')
        
        # Копируем нужные файлы
        shutil.copy2('temp/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe', bin_dir)
        shutil.copy2('temp/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe', bin_dir)
        
        # Удаляем временные файлы
        shutil.rmtree('temp')
        os.remove(zip_file)
    
    elif system == 'Linux':
        # Статические сборки BtbN - работают на любом x86_64 Linux без установки ffmpeg
        url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz'

        # Скачиваем архив
        tar_file = 'ffmpeg.tar.xz'
        download_file(url, tar_file)

        # Распаковываем
        with tarfile.open(tar_file, 'r:xz') as tar_ref:
            tar_ref.extractall('temp')

        # Копируем нужные файлы
        shutil.copy2('temp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg', bin_dir)
        shutil.copy2('temp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe', bin_dir)

        # Делаем файлы исполняемыми (PyInstaller сохраняет права при упаковке
        # и восстановлении onefile-бандла)
        os.chmod(os.path.join(bin_dir, 'ffmpeg'), 0o755)
        os.chmod(os.path.join(bin_dir, 'ffprobe'), 0o755)

        # Удаляем временные файлы
        shutil.rmtree('temp')
        os.remove(tar_file)

    print("FFmpeg и FFprobe успешно скачаны в папку 'bin'")

if __name__ == '__main__':
    download_ffmpeg() 