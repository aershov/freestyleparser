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
        #TODO почему одинаково?
        # Для macOS
        if machine == 'arm64':
            # Apple Silicon
            url_ffprobe = 'https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip'
            url = 'https://evermeet.cx/ffmpeg/getrelease/zip'
        else:
            # Intel
            url_ffprobe = 'https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip'
            url = 'https://evermeet.cx/ffmpeg/getrelease/zip'
        
        # Скачиваем архив
        zip_file = 'ffmpeg.zip'
        download_file(url, zip_file)
        zip_file_ffprobe = 'ffprobe.zip'
        download_file(url_ffprobe, zip_file_ffprobe)

        # Распаковываем
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('temp')
        with zipfile.ZipFile(zip_file_ffprobe, 'r') as zip_ref:
            zip_ref.extractall('temp')


        # Копируем нужные файлы
        shutil.copy2('temp/ffmpeg', os.path.join(bin_dir, 'ffmpeg'))
        shutil.copy2('temp/ffprobe', os.path.join(bin_dir, 'ffprobe'))

        # Делаем файлы исполняемыми
        os.chmod(os.path.join(bin_dir, 'ffmpeg'), 0o755)
        os.chmod(os.path.join(bin_dir, 'ffprobe'), 0o755)
        
        # Удаляем временные файлы
        shutil.rmtree('temp')
        os.remove(zip_file)
        
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
    
    print("FFmpeg и FFprobe успешно скачаны в папку 'bin'")

if __name__ == '__main__':
    download_ffmpeg() 