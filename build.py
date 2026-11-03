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
    # Разделитель src:dest в --add-data на Windows - ';' (':' там часть диска C:)
    data_sep = ';' if system == 'Windows' else ':'

    # Версия: из тега CI (GITHUB_REF_NAME=v1.3.4) или последнего локального
    # тега; попадает в version.txt внутри бандла, app.py показывает её в
    # заголовке - вручную править нигде не нужно
    version = os.environ.get('GITHUB_REF_NAME', '').lstrip('v') \
        if os.environ.get('GITHUB_REF_TYPE') == 'tag' else ''
    if not version:
        try:
            version = subprocess.check_output(
                ['git', 'describe', '--tags', '--abbrev=0'],
                stderr=subprocess.DEVNULL).decode().strip().lstrip('v')
        except Exception:
            version = 'dev'
    with open('version.txt', 'w') as f:
        f.write(version)

    pyinstaller_args = [
        '--name=FreestyleParser',
        '--windowed',
        '--clean',
        f'--add-data=version.txt{data_sep}.',
        f'--add-data=requirements.txt{data_sep}.',
        f'--add-data=bin{data_sep}bin',  # ffmpeg и ffprobe, чтобы не требовались в системе
        f'--add-data=models/yolo11s-seg.pt{data_sep}./models',  # модель YOLO внутри exe
        f'--add-data=models/osnet_x1_0_combineall.pth{data_sep}./models',  # OSNet-веса
        f'--add-data=models/osnet_ain_x1_0_msmt17.pth{data_sep}./models',  # для 'Определять атлетов'
        f'--add-data=icon.ico{data_sep}.',  # иконка окна приложения
        '--collect-all=ultralytics',  # конфиги и данные пакета, без них импорт падает
        '--collect-all=scipy',  # у scipy ленивые импорты (array_api_compat и т.п.) - иначе ModuleNotFoundError в exe
        '--collect-all=supervision',  # ByteTrack в attempts_v2; пакет импортируется только из воркеров - PyInstaller его не видит сам
        '--collect-all=lap',  # C-расширение, нужное supervision.ByteTrack
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
        # onedir, а не onefile: onefile распаковывает весь бандл (torch+cv2,
        # ~1.5 ГБ) в %TEMP% при КАЖДОМ запуске - старт 30-60с, плюс Defender
        # сканирует распакованные файлы. onedir стартует мгновенно;
        # в релиз папка идёт zip'ом
        pyinstaller_args.append('--onedir')
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