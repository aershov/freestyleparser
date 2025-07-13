import os
import platform
import subprocess
import shutil
import sys
from download_ffmpeg import download_ffmpeg

def build_app():
    # Определяем операционную систему
    system = platform.system()
    
    # Очищаем предыдущие сборки
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # Скачиваем ffmpeg и ffprobe, если их нет
    if not os.path.exists('bin'):
        print("Скачиваем FFmpeg и FFprobe...")
        download_ffmpeg()
    
    # Базовые параметры PyInstaller
    pyinstaller_args = [
        '--name=FreestyleParser',
        '--windowed',
        '--add-data=requirements.txt:.',
        '--add-data=bin:bin',  # Добавляем бинарные файлы
        '--add-data=models/yolo11s-seg.pt:./models',  # Добавляем модель
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=yaml',
        'app.py'
    ]
    
    # Добавляем иконку в зависимости от системы
    if system == 'Windows':
        if os.path.exists('icon.ico'):
            pyinstaller_args.append('--icon=icon.ico')
        pyinstaller_args.append('--onefile')
    elif system == 'Darwin':
        if os.path.exists('icon.icns'):
            pyinstaller_args.append('--icon=icon.icns')
        # Для macOS используем --onedir вместо --onefile
        pyinstaller_args.extend([
            '--onedir',
            '--osx-bundle-identifier=com.freestyleparser.app'
        ])
    
    # Запускаем PyInstaller
    try:
        subprocess.run(['pyinstaller'] + pyinstaller_args, check=True)
        
        # Копируем дополнительные файлы
        if system == 'Windows':
            # Копируем DLL файлы для OpenCV
            opencv_dll = os.path.join('venv', 'Lib', 'site-packages', 'cv2', '*.dll')
            if os.path.exists(opencv_dll):
                shutil.copy2(opencv_dll, 'dist')
        elif system == 'Darwin':
            # Подписываем приложение для macOS
            app_path = os.path.join('dist', 'FreestyleParser.app')
            if os.path.exists(app_path):
                try:
                    # Удаляем существующую подпись
                    subprocess.run(['codesign', '--remove-signature', app_path], check=True)
                    # Создаем новую подпись
                    subprocess.run([
                        'codesign',
                        '--force',
                        '--deep',
                        '--sign', '-',  # Используем локальную подпись
                        app_path
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Предупреждение: Не удалось подписать приложение: {e}")
        
        print(f"Сборка завершена успешно. Результат находится в папке 'dist'")
        if system == 'Darwin':
            print(f"Путь к приложению: {os.path.join('dist', 'FreestyleParser.app')}")
        
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при сборке: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_app() 