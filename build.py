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
        '--clean',
        '--add-data=requirements.txt:.',
        '--add-data=bin:bin',  # ffmpeg и ffprobe, чтобы не требовались в системе
        '--add-data=models/yolo11s-seg.pt:./models',  # модель YOLO внутри exe
        '--add-data=icon.ico:.',  # иконка окна приложения
        '--collect-all=ultralytics',  # конфиги и данные пакета, без них импорт падает
        '--collect-all=scipy',  # у scipy ленивые импорты (array_api_compat и т.п.) - иначе ModuleNotFoundError в exe
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
    elif system == 'Linux':
        # Портативный бинарник: один файл без установки зависимостей.
        # Иконку окна Tk берёт из icon.ico во время работы (--icon для ELF нет),
        # ffmpeg/ffprobe уже внутри благодаря --add-data=bin:bin
        pyinstaller_args.append('--onefile')
    elif system == 'Darwin':
        if os.path.exists('icon.icns'):
            pyinstaller_args.append('--icon=icon.icns')
        # Для macOS используем --onedir вместо --onefile
        pyinstaller_args.extend([
            '--onedir',
            '--osx-bundle-identifier=com.freestyleparser.app'
        ])
    
    # Запускаем PyInstaller (через sys.executable - работает и в venv без активации)
    try:
        subprocess.run([sys.executable, '-m', 'PyInstaller'] + pyinstaller_args, check=True)
        
        # Копируем дополнительные файлы
        if system == 'Darwin':
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