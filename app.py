import numpy as np
import tkinter as tk
import calendar
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import subprocess
import datetime
import threading
import yaml
import logging
import shutil
from utils import *
import time

from Components import AthleteWidget
from splitter import process_video, DEFAULT_PROCESSING_PARAMS, load_yolo_model, AttemptInfo
import sys
import traceback

APP_VERSION = "1.0.0.1"

THUMB_X = 180
THUMB_Y = 120

logging.basicConfig(level=logging.INFO)

CANVAS_HEIGHT = 240

MAX_LOG_LINES = 200

def point_in_polygon(point, polygon):
    """
    Проверяет, находится ли точка внутри многоугольника
    Использует алгоритм ray casting
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def create_polygon_mask(polygon_points, width, height):
    """
    Создает маску многоугольника для проверки попадания точек
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_array = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_array], 255)
    return mask

class DatePicker(tk.Toplevel):
    """Простой модальный диалог выбора дня (без внешних зависимостей).
    После wait_window(dialog) результат в dialog.result (datetime.date или None)."""

    MONTHS = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль',
              'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    WEEKDAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

    def __init__(self, parent, initial=None):
        super().__init__(parent)
        self.title("Выберите день")
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        if initial is not None:
            self.year, self.month = initial.year, initial.month
        else:
            today = datetime.date.today()
            self.year, self.month = today.year, today.month

        header = tk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(header, text="◀", width=3, command=lambda: self._shift(-1)).pack(side=tk.LEFT)
        self.month_label = tk.Label(header, width=16, font=('Arial', 11, 'bold'))
        self.month_label.pack(side=tk.LEFT, expand=True)
        tk.Button(header, text="▶", width=3, command=lambda: self._shift(1)).pack(side=tk.LEFT)

        self.grid_frame = tk.Frame(self)
        self.grid_frame.pack(padx=8, pady=4)

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, padx=8, pady=6)
        tk.Button(buttons, text="Сегодня", command=self._pick_today).pack(side=tk.LEFT)
        tk.Button(buttons, text="Отмена", command=self.destroy).pack(side=tk.RIGHT)

        self._render()
        self.grab_set()

    def _render(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.month_label.config(text=f"{self.MONTHS[self.month - 1]} {self.year}")
        for i, wd in enumerate(self.WEEKDAYS):
            tk.Label(self.grid_frame, text=wd, width=4).grid(row=0, column=i)
        today = datetime.date.today()
        for r, week in enumerate(calendar.monthcalendar(self.year, self.month), start=1):
            for c, day in enumerate(week):
                if not day:
                    tk.Label(self.grid_frame, width=4).grid(row=r, column=c)
                    continue
                d = datetime.date(self.year, self.month, day)
                text = f"•{day}" if d == today else str(day)
                tk.Button(self.grid_frame, text=text, width=4,
                          command=lambda dd=d: self._pick(dd)).grid(row=r, column=c)

    def _shift(self, delta):
        m = self.month + delta
        if m < 1:
            m, self.year = 12, self.year - 1
        elif m > 12:
            m, self.year = 1, self.year + 1
        self.month = m
        self._render()

    def _pick(self, d):
        self.result = d
        self.destroy()

    def _pick_today(self):
        self._pick(datetime.date.today())


class FreestyleParserApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Freestyle Parser {APP_VERSION}")
        self.root.geometry("1605x900")

        # Цвет фона кнопок по умолчанию: 'SystemButtonFace' существует только
        # на Windows, на Linux/macOS Tk не знает такого имени цвета,
        # поэтому берём фактическое значение из Tk
        probe_button = tk.Button(root)
        self.button_face_color = probe_button.cget('bg')
        probe_button.destroy()

        # Инициализируем переменные
        current_date = datetime.date.today()
        formatted_date = current_date.strftime("%Y-%m-%d")
        self._app_folder = os.path.expanduser("~/.FreestyleParser")
        if not os.path.exists(self._app_folder):
            os.makedirs(self._app_folder, exist_ok=True)
        # Рабочая папка: последняя использованная, иначе папка текущего дня
        last_folder = self._load_last_folder()
        if last_folder:
            self.output_folder = last_folder
        else:
            self.output_folder = os.path.expanduser(f"~/Desktop/FreestyleParser/{formatted_date}")
        self.processing_config_file = os.path.join(self.output_folder, "processing.yaml")
        #TODO почистить это всё (
        self.roi = None  # Теперь это будет список точек многоугольника в процентах
        self.roi_points = []  # Точки многоугольника в пикселях canvas
        self.roi_mode = "polygon"  # "rectangle" или "polygon"
        self.drawing_polygon = False
        self.logger = None
        self.selected_files = []
        self.processing = False
        # Раздельные флаги задач: скан и нарезка кандидатов могут идти одновременно
        # (self.processing - общий флаг "идёт работа", сбрасывается кнопкой Остановить)
        self.scan_running = False
        self.cut_running = False
        self._candidates_lock = threading.RLock()
        self.need_update_attempts = False
        self.athlete_mapping = {}  # Инициализируем пустой словарь
        self._thumb_model = None   # кеш YOLO для умных превью
        self._embedder = None      # кеш OSNet-эмбеддера для профилей из клипов
        self.attempt_assignments = {}  # авто-привязки: {attempt: {athlete, sim, second}}
        self._assignments_lock = threading.Lock()
        self._registry_cache = None   # глобальный реестр атлетов (~/.FreestyleParser)
        self._base_banks_cache = None  # базовые банки (read-only, кеш сессии)
        self.athlete_segments = []    # сегменты скана атлетов (без нарезки)
        self._segment_profiles = {}   # {key: {'vest', 'boat'}} профили сегментов
        self.need_update_segments = False
        self.need_update_files = False
        self.athlete_widgets = []
        self.filter_var = tk.StringVar(value="Все")
        self.selected_attempts = set()  # Множество выбранных попыток
        self.attempt_checkboxes = {}  # Словарь для хранения чекбоксов попыток
        self.attempt_ratings = {}  # Словарь для хранения рейтингов попыток: {attempt: {'up': bool, 'down': bool}}
        self.active_rating_filters = set()  # Множество активных фильтров по рейтингу
        self.processing_params = DEFAULT_PROCESSING_PARAMS.copy()
        self._progress_percent = 0.0
        self.slowmo_var = None  # tk.BooleanVar, создаётся в setup_ui
        self._active_slowmo_factor = 1  # коэффициент замедления текущей обработки
        self._audio_stream_cache = {}  # video_path -> bool (есть ли аудио-дорожка)
        self._file_duration_cache = {}  # video_path -> длительность файла, с
        self.save_now_var = None     # [ ] Сразу сохранять (создаётся в setup_ui)
        self.athletes_var = None     # [ ] Определять атлетов (создаётся в setup_ui)
        self.candidates = []  # кандидаты попыток ручного режима: {file,start,end,thumbnail,selected,status}
        self.candidate_vars = {}  # ключ кандидата -> tk.BooleanVar чекбокса
        self.candidates_file = None  # путь candidates.yaml, обновляется в on_output_folder_changed
        self._scan_found = 0

        # self.on_output_folder_changed()

        # Создаем интерфейс
        self.setup_ui()
        self._setup_window_icon()
        self._restore_selected_files()
        if os.path.exists(self.output_folder):
            self.on_output_folder_changed()
        else:
            # Папки нет - явно показываем это в интерфейсе
            self.update_folder_status()
            print(f"Рабочая папка не существует: {self.output_folder} - "
                  f"выберите день (📅) или папку (Изменить)")
        # Запускаем периодическое обновление
        self.root.after(1000, self.periodic_update)

    def _setup_window_icon(self):
        """Иконка окна: icon.ico рядом с app.py (в exe - из распакованного бандла)"""
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.ico')
        try:
            if os.path.exists(icon_path):
                self._icon_image = ImageTk.PhotoImage(Image.open(icon_path))
                self.root.iconphoto(True, self._icon_image)
        except Exception as e:
            print(f"Не удалось установить иконку окна: {e}")

    def periodic_update(self):
        """Периодическое обновление интерфейса"""
        if self.need_update_attempts:
            self.update_attempt_thumbnails()
            self.need_update_attempts = False
        if self.need_update_segments:
            self.need_update_segments = False
            self.update_segments_panel()
        if self.need_update_files:
            self.need_update_files = False
            self._render_file_list()
        if self.processing:
            self.progress_var.set(min(self._progress_percent, 100.0))
        self.root.after(1000, self.periodic_update)

    def on_attempt_drag_end(self, event, attempt):
        """Обработчик окончания перетаскивания попытки"""
        # Отменяем таймер перетаскивания
        if hasattr(self, 'drag_timer'):
            self.root.after_cancel(self.drag_timer)
            delattr(self, 'drag_timer')

        # Удаляем окно перетаскивания
        if hasattr(self, 'drag_window'):
            self.drag_window.destroy()
            delattr(self, 'drag_window')

        # Получаем координаты курсора
        x = event.x_root
        y = event.y_root

        # Проверяем, находится ли курсор над кнопкой "+ Новый атлет"
        button_x = self.button_add_athlete.winfo_rootx()
        button_y = self.button_add_athlete.winfo_rooty()
        button_width = self.button_add_athlete.winfo_width()
        button_height = self.button_add_athlete.winfo_height()

        if (button_x <= x <= button_x + button_width and
                button_y <= y <= button_y + button_height):
            # Если курсор над кнопкой "+ Новый атлет", отвязываем попытку
            self.assign_attempt_to_athlete(attempt, None)
            self.log(f"Попытка {attempt} отвязана от атлета")
            return

        # Проверяем, находится ли курсор над виджетом атлета
        for athlete_widget in self.athlete_widgets:
            # Получаем координаты виджета атлета
            athlete_x = athlete_widget.winfo_rootx()
            athlete_y = athlete_widget.winfo_rooty()
            athlete_width = athlete_widget.winfo_width()
            athlete_height = athlete_widget.winfo_height()

            # Проверяем, находится ли курсор над виджетом атлета
            if (athlete_x <= x <= athlete_x + athlete_width and
                    athlete_y <= y <= athlete_y + athlete_height):
                # Получаем имя атлета
                athlete_name = athlete_widget.name

                # Обновляем маппинг
                self.assign_attempt_to_athlete(attempt, athlete_name)
                self.log(f"Попытка {attempt} привязана к атлету {athlete_name}")
                return

    def on_filter_change(self, event):
        """Обработчик изменения фильтра"""
        self.need_update_attempts = True  # Устанавливаем флаг обновления

    def _settings_file(self):
        return os.path.join(self._app_folder, "settings.yaml")

    def _load_last_folder(self):
        """Последняя рабочая папка из настроек (если ещё существует)"""
        try:
            f = self._settings_file()
            if os.path.exists(f):
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                p = data.get('last_output_folder')
                if p and os.path.isdir(p):
                    return p
        except Exception:
            pass
        return None

    def _save_settings(self):
        """Сохраняет последнюю рабочую папку"""
        try:
            with open(self._settings_file(), "w", encoding="utf-8") as f:
                yaml.dump({'last_output_folder': self.output_folder}, f, allow_unicode=True)
        except Exception as e:
            print(f"Не удалось сохранить настройки: {e}")

    def _restore_selected_files(self):
        """Восстанавливает выбор файлов этого дня (processing.yaml в папке дня):
        недоделанная работа видна сразу; отсутствующие файлы - красным."""
        try:
            with open(self.processing_config_file, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            files = (config.get('processing-config') or {}).get('selected_files') or []
        except Exception:
            files = []
        self.selected_files = [f for f in files if isinstance(f, str)]
        if not self.selected_files:
            return
        self._render_file_list()
        existing = [f for f in self.selected_files if os.path.exists(f)]
        if existing:
            try:
                self.current_video_for_roi = existing[0]
                self.current_frame_position = 0.5
                self.create_canvas_from_video(existing[0])
                factor = detect_slowmo_factor(existing[0])
                self._set_slowmo_ui(factor)
            except Exception as e:
                self.log(f"Не удалось восстановить предпросмотр: {e}", logging.WARNING)
        missing = len(self.selected_files) - len(existing)
        self.log(f"Восстановлен выбор файлов: {len(self.selected_files)}"
                 + (f" (нет на диске: {missing} - отмечены красным)" if missing else ""))

    def _render_file_list(self):
        """Список выбранных файлов с отметками состояния:
        отсутствует на диске - красным с суффиксом; в папке с частичной
        работой (есть отметки/кандидаты/сегменты) необработанные файлы -
        красным. В чистой папке ничего не краснеет."""
        if not hasattr(self, 'listbox_files'):
            return
        self.listbox_files.delete(0, tk.END)
        folder_ready = os.path.isdir(self.output_folder)
        partial_work = folder_ready and self._folder_has_work()
        processed = self._load_processed()
        for i, f in enumerate(self.selected_files):
            base = os.path.basename(f)
            exists = os.path.exists(f)
            self.listbox_files.insert(tk.END,
                                      base + ('' if exists else '  (нет файла)'))
            if not exists or (partial_work and base not in processed
                              and not self._file_is_done(base)):
                self.listbox_files.itemconfig(i, fg='#d32f2f')

    def _folder_has_work(self):
        """Есть ли в папке следы обработки (значит, недоделанные файлы -
        это именно недоделка, а не чистый лист)."""
        if self.candidates or self.athlete_segments:
            return True
        return bool(self._load_processed())

    def _file_is_done(self, base):
        """Файл уже обработан в этой папке: есть отметка processed.yaml,
        кандидаты или сегменты из этого файла."""
        for c in self.candidates:
            if os.path.basename(str(c.get('file', ''))) == base:
                return True
        for s in self.athlete_segments:
            if os.path.basename(str(s.get('file', ''))) == base:
                return True
        return False

    # === Отметки обработанных файлов (папка дня, processed.yaml) ===

    def _processed_file(self):
        return os.path.join(self.output_folder, 'processed.yaml')

    def _load_processed(self):
        """{basename исходника: {'strategy', 'when'}} - что уже обработано
        в текущей папке (любой стратегией)."""
        f = self._processed_file()
        if os.path.exists(f):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    return yaml.safe_load(fh) or {}
            except Exception:
                pass
        return {}

    def _log_processing_summary(self, strategy, files, t0, completed):
        """Итог прогона в лог: тип процессинга, длительность, число файлов."""
        mins, secs = divmod(max(0, int(time.time() - t0)), 60)
        state = "завершён" if completed else "ПРЕРВАН"
        self.log(f"Итог [{strategy}]: {state}, время {mins} мин {secs:02d} с, "
                 f"файлов: {len(files)}")

    def _mark_processed(self, video_path, strategy):
        """Отмечает исходник обработанным (после успешного прогона файла)."""
        data = self._load_processed()
        data[os.path.basename(video_path)] = {
            'strategy': strategy,
            'when': datetime.datetime.now().isoformat(timespec='seconds')}
        try:
            with open(self._processed_file(), 'w', encoding='utf-8') as fh:
                yaml.dump(data, fh, allow_unicode=True)
        except Exception as e:
            self.log(f"Не удалось записать отметку обработки: {e}", logging.WARNING)
        self.need_update_files = True

    def _set_output_folder(self, folder, log_message=None):
        """Переключает рабочую папку: UI, состояние, настройки"""
        self.output_folder = folder
        self.entry_output.config(state='normal')
        self.entry_output.delete(0, tk.END)
        self.entry_output.insert(0, os.path.basename(folder))
        self.entry_output.config(state='readonly')
        self._save_settings()
        self.on_output_folder_changed()
        self.log(log_message or f"Рабочая папка: {folder}")

    def update_folder_status(self):
        """Показывает состояние рабочей папки: создана или нет"""
        if not hasattr(self, 'folder_status_label'):
            return
        if os.path.isdir(self.output_folder):
            self.folder_status_label.config(text="✓ папка создана", fg="#2e7d32")
            self.entry_output.config(bg='white')
        elif self.selected_files:
            # файлы уже выбраны - папка будет создана при сохранении конфига/обработке
            self.folder_status_label.config(
                text="⏳ папка дня будет создана при запуске обработки", fg="#ef6c00")
            self.entry_output.config(bg='#fff3e0')
        else:
            self.folder_status_label.config(
                text="✗ папка не создана — выберите 📅 день или папку", fg="#c62828")
            self.entry_output.config(bg='#ffdddd')

    def choose_day_folder(self):
        """Выбор папки дня через календарь: ~/Desktop/FreestyleParser/YYYY-MM-DD"""
        dialog = DatePicker(self.root, datetime.date.today())
        self.root.wait_window(dialog)
        d = dialog.result
        if d is None:
            return
        folder = os.path.expanduser(f"~/Desktop/FreestyleParser/{d.isoformat()}")
        self._set_output_folder(folder, f"Рабочая папка дня: {folder}")

    def on_filter_change(self, event):
        """Обработчик изменения фильтра"""
        self.need_update_attempts = True  # Устанавливаем флаг обновления

    @property
    def canvas_width(self):
        w = self.canvas.winfo_width()
        return w if w > 1 else 480

    def start_processing(self):
        """Запускает обработку видео"""

        self._sync_params_from_ui()
        # на случай, если папка по умолчанию, надо добавить проверку и делать это только если output_folder не exists
        self.save_processing_config()
        self.on_output_folder_changed()
        if not self.selected_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите файлы для обработки")
            return

        if not self.roi:
            messagebox.showwarning("Предупреждение", "Сначала выберите область интереса")
            return
        self.processing = True
        self._active_slowmo_factor = self._ui_slowmo_factor()
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        self.log(f"Запускаем процессинг файлов с roi={self.roi}")
        # Запускаем обработку в отдельном потоке
        strategy = self._strategy()
        if strategy == 'manual':
            self._start_worker(self.process_scan)
        elif strategy == 'attempts':
            self._start_worker(self.process_attempts)
        elif strategy == 'athletes':
            self._start_worker(self.process_athlete_scan)
        else:
            self._start_worker(self.process_videos)

    def _strategy(self):
        """Стратегия из чекбоксов: 'auto'|'manual'|'attempts'|'athletes'.
        Внутренние коды сохранены для совместимости processing.yaml."""
        if not hasattr(self, 'save_now_var') or self.save_now_var is None:
            return 'auto'
        save_now = self.save_now_var.get()
        athletes = self.athletes_var.get()
        if save_now and athletes:
            return 'attempts'
        if save_now:
            return 'auto'
        return 'athletes' if athletes else 'manual'

    def on_strategy_change(self):
        """Чекбоксы стратегии меняются - сохраняем конфиг"""
        self.save_processing_config()

    def _start_worker(self, target):
        """Запускает поток обработки со страховкой: любая ошибка внутри воркера
        логируется и гарантированно возвращает кнопки в активное состояние."""
        def run():
            try:
                target()
            except Exception:
                self.log("Ошибка потока обработки: " + traceback.format_exc(),
                         logging.ERROR)
                self.processing = False
                self.root.after(0, lambda: self._after_worker_ui(False))
        threading.Thread(target=run, daemon=True).start()

    def _video_has_audio(self, video_path):
        """Проверяет наличие аудио-дорожки (с кэшем по файлу)"""
        if video_path not in self._audio_stream_cache:
            self._audio_stream_cache[video_path] = has_audio_stream(video_path)
        return self._audio_stream_cache[video_path]

    def _get_file_duration(self, video_path):
        """Длительность видеофайла в секундах (с кэшем по файлу)"""
        if video_path not in self._file_duration_cache:
            try:
                d = float(subprocess.check_output(
                    [get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'csv=p=0', video_path], timeout=30).decode().strip())
            except Exception:
                d = 0.0
            self._file_duration_cache[video_path] = d
        return self._file_duration_cache[video_path]

    def _estimate_output_bytes(self, video_path, duration_real):
        """Оценка размера нарезанного видео по битрейту исходника.
        Без слоумо - потоковое копирование: размер = битрейт * длительность.
        Слоумо - перекодирование x264 crf18: по замеру на реальном 4K-исходнике
        выход получается ~в 3 раза меньше битрейта камеры (эмпирический /3)."""
        slowmo = self._ui_slowmo_factor()
        try:
            size = os.path.getsize(video_path)
        except OSError:
            return 0
        fdur = self._get_file_duration(video_path)
        if size <= 0 or fdur <= 0 or duration_real <= 0:
            return 0
        return size / fdur * duration_real * slowmo / (3.0 if slowmo > 1 else 1.0)

    @staticmethod
    def _fmt_size(nbytes):
        """Человекочитаемый размер: КБ/МБ/ГБ"""
        if nbytes >= 1024 ** 3:
            return f"{nbytes / 1024 ** 3:.1f} ГБ"
        if nbytes >= 1024 ** 2:
            return f"{nbytes / 1024 ** 2:.0f} МБ"
        return f"{nbytes / 1024:.0f} КБ"

    def on_attempt_file_created(self, attempt):
        ## TODO analyze best_frame
        cv2.imwrite(os.path.join(self.output_folder, f"{attempt.number:04d}.jpg"), attempt.best_frame)
        # cv2.imwrite(os.path.join(self.output_folder, f"{attempt.number:04d}-base.jpg"), attempt.base_frame)
        # person_roi_frame = extract_athlete_difference(attempt.base_frame, attempt.person_frame)
        # colors = self.get_colors_from_athlete(attempt.base_frame, attempt.person_frame, None, 4, 8)
        # pers_file_path = os.path.join(self.output_folder, f"{attempt.number:04d}-person-{colors}.jpg")
        # cv2.imwrite(pers_file_path, attempt.person_frame)
        # self.log(f"Colors {colors}")
        output_file = os.path.join(self.output_folder, f"{attempt.number:04d}.mp4")
        duration = attempt.end - attempt.start
        if duration <= 0:
            self.log(f"Пропускаем попытку {attempt.number}: start={attempt.start:.2f}s >= end={attempt.end:.2f}s")
            return self.processing
        slowmo = self._active_slowmo_factor
        if slowmo > 1:
            # Слоумо: кадры в контейнере идут с интервалом 1/24с, а поток H.264
            # содержит 96 уникальных кадров в секунду. Сжимаем таймстампы в
            # slowmo раз (-itsscale) и выставляем timescale = 96000, чтобы
            # плеер воспроизводил попытку на реальной скорости (96 к/с)
            # без перекодирования. -ss/-t задаются в реальных секундах.
            ts = int(slowmo * 24) * 1000  # 96000 для 96 fps
            self.log(f"Слоумо 1/{slowmo}: remux {attempt.start:.2f}s..{attempt.start + duration:.2f}s "
                     f"(реальных), stream copy (itsscale={1 / slowmo}, timescale={ts})")
            cmd = [get_ffmpeg_path(),
                   "-itsscale", f"{1 / slowmo}",
                   "-i", attempt.source_video,
                   "-ss", f"{attempt.start:.3f}",
                   "-t", f"{duration:.3f}",
                   "-c", "copy",
                   "-video_track_timescale", str(ts),
                   "-map", "0:v:0",
                   "-movflags", "+faststart",
                   "-y", output_file]
            if self._video_has_audio(attempt.source_video):
                cmd += ["-map", "0:a:0?"]
        else:
            cmd = [get_ffmpeg_path(), "-ss", str(attempt.start), "-i", attempt.source_video,
                   "-t", str(duration), "-c", "copy", "-y", output_file]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                self.log(f"Ошибка ffmpeg для попытки {attempt.number}: {result.stderr[-500:]}", logging.ERROR)
                return self.processing
        except subprocess.TimeoutExpired:
            self.log(f"Таймаут ffmpeg для попытки {attempt.number} (>300с)", logging.ERROR)
            return self.processing
        except Exception as e:
            self.log(f"Ошибка запуска ffmpeg для попытки {attempt.number}: {e}", logging.ERROR)
            return self.processing

        # Контрольная проверка: длительность выхода должна совпадать с ожидаемой.
        # Расхождение означает, что вырезка попала не туда (например, неверный
        # режим слоумо или кривые таймстампы исходника).
        try:
            out_duration = float(subprocess.check_output(
                [get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', output_file]).decode().strip())
            if abs(out_duration - duration) > max(1.0, duration * 0.2):
                self.log(f"ВНИМАНИЕ: попытка {attempt.number}: длительность результата "
                         f"{out_duration:.2f}s сильно отличается от ожидаемой {duration:.2f}s - "
                         f"проверьте режим слоумо", logging.WARNING)
        except Exception:
            pass

        # self.log(f"Saved attempt {attempt.number} from {attempt.start:.2f}s to {attempt.end:.2f}s (duration: {attempt.duration():.2f}s)")

        self.need_update_attempts = True
        self.log(f"Видео попытки готово: {output_file}")
        # сообщаем, продолжать или нет (если была отмена кнопкой - то остановимся)
        return self.processing

    def get_colors_from_athlete(self, frame_bg, frame_with, bbox=None, k=5, threshold=10):
        roi = extract_athlete_difference(frame_bg, frame_with, bbox)
        if roi is None:
            return []

        # --- Фильтруем пустые/неподходящие регионы ---
        non_zero_pixels = cv2.countNonZero(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0 or non_zero_pixels / total_pixels < 0.05:
            print("[INFO] ROI слишком мал или пуст")
            return []

        # --- Анализируем цвета только в этой области ---
        dominant_colors = get_dominant_colors(roi, k, threshold)

        return dominant_colors


    def process_videos(self):
        """Обрабатывает видео в отдельном потоке (авто-режим: нарезать всё)"""
        t0 = time.time()
        try:
            next_attempt_number = get_next_attempt_number(self.output_folder)
            process_video(self.selected_files, self.roi, lambda attempt: self.on_attempt_file_created(attempt),
                          next_attempt_number, params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=self._active_slowmo_factor)
            self.log(f"Файлы {self.selected_files} успешно обработаны.")
            if self.processing:
                for f in self.selected_files:
                    self._mark_processed(f, 'auto')
            self._log_processing_summary("Нарезка всех попыток (авто)",
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.log(f"Ошибка обработки: {e} {error_trace}")
        finally:
            self.scan_running = False
            self.processing = self.cut_running
            self.root.after(0, lambda: self._after_worker_ui(self.processing))

    def process_scan(self):
        """Сканирует файлы и собирает кандидатов попыток (ручной режим).
        Кандидаты неточные (шаг SCAN_INTERVAL), нарезаются после подтверждения.
        Повторный скан ДОБАВЛЯЕТ новых кандидатов, не трогая существующие
        (уже отмеченные/нарезанные не пропадают) - удобно, если с первыми
        параметрами что-то не нашлось. Может идти одновременно с нарезкой
        кандидатов (кнопка 'Нарезать выбранное')."""
        self.scan_running = True
        t0 = time.time()
        try:
            existing = list(self.candidates)
            if existing:
                self.log(f"Пересканирование: {len(existing)} существующих кандидатов сохраняются, "
                         f"новые добавляются без дублей")
            process_video(self.selected_files, self.roi, self.on_candidate_found,
                          1, params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=self._active_slowmo_factor,
                          scan_mode=True)
            self.save_candidates()
            if self.processing:
                for f in self.selected_files:
                    self._mark_processed(f, 'manual')
            self._log_processing_summary("Скан кандидатов (ручной)",
                                         self.selected_files, t0, self.processing)
            pending = [c for c in self.candidates if c['status'] == 'pending']
            self.log(f"Сканирование завершено: всего кандидатов {len(pending)}. "
                     f"Снимите галочки с ненужных попыток и нажмите 'Нарезать выбранное'.")
        except Exception as e:
            self.log(f"Ошибка сканирования: {e} {traceback.format_exc()}", logging.ERROR)
        finally:
            self.scan_running = False
            self.processing = self.cut_running
            still_busy = self.processing
            self.root.after(0, lambda: self._after_worker_ui(still_busy))
            self.root.after(0, self.update_attempt_thumbnails)

    def _after_worker_ui(self, still_busy):
        """Обновляет кнопки после завершения задачи; если работают другие - не трогаем."""
        if not still_busy:
            self.button_process.config(state=tk.NORMAL)
            self.button_stop.config(state=tk.DISABLED)
        self.update_cut_button_state()

    # === Режим 'Попытки (атриб.)': сплиттер попыток + автоатрибуция ===

    ATTEMPT_SIM_THRESHOLD = 0.88   # минимальная похожесть для автоатрибуции
    ATTEMPT_SIM_MARGIN = 0.03      # отрыв от второго кандидата
    W_REID, W_BOAT = 0.7, 0.3      # вклад OSNet и лодки в похожесть

    def _roi_bbox_percentages(self):
        """BBox ROI в процентах (L, T, R, B) из прямоугольника или полигона."""
        if len(self.roi) == 4:
            return tuple(self.roi)
        xs = [p[0] for p in self.roi]
        ys = [p[1] for p in self.roi]
        return (min(xs), min(ys), max(xs), max(ys))

    def _next_attempt_number(self):
        """Следующий свободный номер попытки в папке дня."""
        nums = []
        if os.path.exists(self.output_folder):
            for f in os.listdir(self.output_folder):
                if f.endswith('.mp4') and f[:4].isdigit():
                    nums.append(int(f[:4]))
        return max(nums, default=0) + 1

    def _load_profile_store(self):
        """Профили попыток дня: {basename: {'vest': vec, 'boat': vec}}."""
        store_file = os.path.join(self.output_folder, 'attempt_profiles.npz')
        if not os.path.exists(store_file):
            return {}
        d = np.load(store_file, allow_pickle=True)
        store = {}
        for i, name in enumerate(d['names']):
            store[str(name)] = {'vest': d['vest'][i], 'boat': d['boat'][i]}
        return store

    def _save_profile_store(self, store):
        store_file = os.path.join(self.output_folder, 'attempt_profiles.npz')
        names = list(store.keys())
        np.savez(store_file,
                 names=np.array(names),
                 vest=np.array([store[n]['vest'] for n in names]),
                 boat=np.array([store[n]['boat'] for n in names]))

    # === Глобальный реестр атлетов (~/.FreestyleParser, живёт между папками) ===

    REGISTRY_SAMPLES_MAX = 20  # сэмплов профилей на атлета

    def _registry_file(self):
        return os.path.join(self._app_folder, 'athlete_registry.npz')

    def _load_registry(self):
        """Реестр: {имя: {'vest': [векторы 1664], 'boat': [векторы 18]}}.
        Кешируется; пополняется подтверждениями попыток и сегментов."""
        if self._registry_cache is not None:
            return self._registry_cache
        reg = {}
        f = self._registry_file()
        if os.path.exists(f):
            try:
                d = np.load(f, allow_pickle=True)
                for i, name in enumerate(d['names']):
                    name = str(name)
                    reg.setdefault(name, {'vest': [], 'boat': []})
                    reg[name]['vest'].append(d['vest'][i])
                    reg[name]['boat'].append(d['boat'][i])
            except Exception as e:
                self.log(f"Ошибка загрузки реестра атлетов: {e}", logging.WARNING)
        self._registry_cache = reg
        return reg

    def _save_registry(self, reg):
        f = self._registry_file()
        try:
            names, vests, boats = [], [], []
            for name, s in reg.items():
                for v, b in zip(s['vest'], s['boat']):
                    names.append(name)
                    vests.append(v)
                    boats.append(b)
            if names:
                np.savez(f, names=np.array(names),
                         vest=np.array(vests), boat=np.array(boats))
            elif os.path.exists(f):
                os.remove(f)
            self._registry_cache = reg
        except Exception as e:
            self.log(f"Ошибка сохранения реестра атлетов: {e}", logging.ERROR)

    def _registry_add_sample(self, name, vest, boat):
        """Добавляет профиль подтверждённой попытки/сегмента в реестр."""
        import attempts_v2
        if vest is None or float(np.linalg.norm(vest[attempts_v2.VEST_DIM:])) < 1e-6:
            return
        reg = self._load_registry()
        s = reg.setdefault(name, {'vest': [], 'boat': []})
        s['vest'].append(np.asarray(vest, dtype=float))
        s['boat'].append(np.asarray(boat if boat is not None else np.zeros(18),
                                    dtype=float))
        if len(s['vest']) > self.REGISTRY_SAMPLES_MAX:
            s['vest'] = s['vest'][-self.REGISTRY_SAMPLES_MAX:]
            s['boat'] = s['boat'][-self.REGISTRY_SAMPLES_MAX:]
        self._save_registry(reg)

    def _registry_banks(self):
        """Банки из реестра: СЫРЫЕ сэмплы по атлетам (списки векторов).
        Усреднение не делаем: сэмплы разных скинов (лето/зима, куртки)
        в среднем не совпадут ни с чем - сравнение идёт max по сэмплам."""
        banks = {}
        for name, s in self._load_registry().items():
            if not s['vest']:
                continue
            banks[name] = {
                'vest': [v / max(np.linalg.norm(v), 1e-9) for v in s['vest']],
                'boat': [b / max(b.sum(), 1e-9) for b in s['boat']]}
        return banks

    # === Сегменты скана атлетов (без нарезки) ===

    def _segments_file(self):
        return os.path.join(self.output_folder, 'segments.yaml')

    def _segment_store_file(self):
        return os.path.join(self.output_folder, 'segment_profiles.npz')

    def _load_segments(self):
        self.athlete_segments = []
        self._segment_profiles = {}
        f = self._segments_file()
        if os.path.exists(f):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh) or {}
                self.athlete_segments = data.get('segments', [])
            except Exception as e:
                self.log(f"Ошибка загрузки segments: {e}", logging.WARNING)
        sf = self._segment_store_file()
        if os.path.exists(sf):
            try:
                d = np.load(sf, allow_pickle=True)
                self._segment_profiles = {
                    str(n): {'vest': d['vest'][i], 'boat': d['boat'][i]}
                    for i, n in enumerate(d['names'])}
            except Exception as e:
                self.log(f"Ошибка загрузки профилей сегментов: {e}", logging.WARNING)

    def _save_segments(self):
        try:
            with open(self._segments_file(), 'w', encoding='utf-8') as fh:
                yaml.dump({'segments': self.athlete_segments}, fh,
                          allow_unicode=True)
        except Exception as e:
            self.log(f"Ошибка сохранения segments: {e}", logging.ERROR)

    def _save_segment_profiles(self):
        names = list(self._segment_profiles)
        try:
            if names:
                np.savez(self._segment_store_file(),
                         names=np.array(names),
                         vest=np.array([self._segment_profiles[n]['vest']
                                        for n in names]),
                         boat=np.array([self._segment_profiles[n]['boat']
                                        for n in names]))
            elif os.path.exists(self._segment_store_file()):
                os.remove(self._segment_store_file())
        except Exception as e:
            self.log(f"Ошибка сохранения профилей сегментов: {e}", logging.ERROR)

    def _base_banks(self):
        """Базовые банки ~/.FreestyleParser/banks/*.npz (read-only, формат
        как у registry.npz: names/vest/boat, строки-сэмплы). Возвращаем
        СЫРЫЕ сэмплы списками - сравнение max по сэмплам (скины/сезоны).
        Кешируется на сессию (банки read-only); вызывается на каждую плитку."""
        if getattr(self, '_base_banks_cache', None) is not None:
            return self._base_banks_cache
        samples = {}
        bdir = os.path.join(self._app_folder, 'banks')
        if os.path.isdir(bdir):
            for fn in sorted(os.listdir(bdir)):
                if not fn.endswith('.npz'):
                    continue
                try:
                    d = np.load(os.path.join(bdir, fn), allow_pickle=True)
                    for i, n in enumerate(d['names']):
                        samples.setdefault(str(n), []).append(
                            (d['vest'][i], d['boat'][i]))
                except Exception as e:
                    self.log(f"Ошибка загрузки банка {fn}: {e}", logging.WARNING)
        banks = {}
        for name, s in samples.items():
            banks[name] = {
                'vest': [v / max(np.linalg.norm(v), 1e-9) for v, _ in s],
                'boat': [b / max(b.sum(), 1e-9) for _, b in s]}
        self._base_banks_cache = banks
        return banks

    def _athlete_banks(self, store):
        """Банки атлетов по приоритету: подтверждённый маппинг папки >
        личный реестр > базовые банки. Локальные затирают базовые
        при совпадении имён."""
        banks = {}
        for athlete, files in self.athlete_mapping.items():
            entries = [store[os.path.basename(f)] for f in files
                       if os.path.basename(f) in store]
            if not entries:
                continue
            # каждая попытка дня - отдельный сэмпл банка
            banks[athlete] = {
                'vest': [e['vest'] / max(np.linalg.norm(e['vest']), 1e-9)
                         for e in entries],
                'boat': [e['boat'] / max(e['boat'].sum(), 1e-9)
                         for e in entries]}
        merged = dict(self._base_banks())
        merged.update(self._registry_banks())
        merged.update(banks)
        return merged

    def _attempt_athlete_sim(self, profile, bank):
        """Похожесть попытки и банка: MAX по сэмплам банка (скины, сезоны,
        ракурсы). Сэмпл = OSNet+жилет косинус + совпадение бина лодки."""
        # errstate: spurious overflow/divide-by-zero warning от Accelerate BLAS
        # (macOS) на матмуле таких размеров - результат корректен, проверено
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            vests = bank['vest'] if isinstance(bank['vest'], list) \
                else [bank['vest']]
            boats = bank['boat'] if isinstance(bank['boat'], list) \
                else [bank['boat']]
            best, best_reid = -1.0, 0.0
            for v, b in zip(vests, boats):
                s_reid = float(profile['vest'] @ v)
                b_hist = profile['boat']
                if b_hist.sum() > 0:
                    bp = b_hist / b_hist.sum()
                    top = int(np.argmax(bp))
                    bt = int(np.argmax(b))
                    dist = min(abs(top - bt), 18 - abs(top - bt))
                    s_boat = 1.0 if dist <= 1 else 0.0
                    s = self.W_REID * s_reid + self.W_BOAT * s_boat
                else:
                    s = s_reid * self.W_REID / (self.W_REID + self.W_BOAT)
                if s > best:
                    best, best_reid = s, s_reid
            return best, best_reid

    def process_attempts(self):
        """Стратегия 'Попытки (атриб.)': сплиттер попыток (YOLO-seg + трекинг),
        нарезка попыток и автоатрибуция по банкам подтверждённых атлетов."""
        t0 = time.time()
        try:
            import attempts_v2
            roi_bbox = self._roi_bbox_percentages()
            store = self._load_profile_store()
            # Удаляем записи, для которых нет .mp4 файлов (старые попытки из прошлых запусков)
            stale = [n for n in store if not os.path.exists(os.path.join(self.output_folder, n))]
            if stale:
                self.log(f"Удаляю {len(stale)} устаревших записей из store: {', '.join(stale)}")
                for n in stale:
                    del store[n]
            # мёртвые привязки маппинга - до построения банков и нумерации,
            # иначе новая попытка с тем же номером унаследует чужую привязку
            self._prune_athlete_mapping()
            banks = self._athlete_banks(store)
            if banks:
                self.log(f"Банки атлетов: {', '.join(banks.keys())}")
            else:
                self.log("Банков атлетов пока нет - попытки будут нарезаны без атрибуции "
                         "(после привязки попыток к атлетам атрибуция заработает)")
            number = self._next_attempt_number()
            start_pad = float(self.processing_params.get('attempt_start_padding', 2))
            # запас на вход атлета в ROI, как в точном анализе старого пайплайна
            margin = float(self.processing_params.get('scan_interval', 2.5)) + 1.0
            end_pad = float(self.processing_params.get('attempt_end_padding', 0.5))
            found_in_file = [0]

            def on_attempt_found(r):
                """Прогрессивная нарезка: попытка режется сразу после закрытия."""
                nonlocal number
                # preroll = start_pad (2с до входа) — даёт каякеру появиться во 2й секунде
                preroll = max(0.0, r['start'] - start_pad)
                # thumbnail: кадр, где атлет ближе всего к центру ROI
                # (attempts_v2 считает thumb_t по дистанции bbox-центра)
                thumb_t = r.get('thumb_t') or r['start']
                src = cv2.VideoCapture(r['source'])
                src.set(cv2.CAP_PROP_POS_MSEC, thumb_t * self._active_slowmo_factor * 1000)
                ok, best_frame = src.read()
                src.release()
                if not ok:
                    best_frame = np.zeros((360, 640, 3), np.uint8)
                info = AttemptInfo(r['source'], preroll,
                                   r['end'] + end_pad, number,
                                   best_frame, None, None, None)
                self.on_attempt_file_created(info)
                out_name = f"{number:04d}.mp4"
                number += 1
                found_in_file[0] += 1
                self.log(f"  Попытка {out_name}: {r['start']:.1f}-{r['end']:.1f}с "
                         f"(лодка {r['boat_top'] * 10 if r['boat_top'] is not None else '-'}°)")
                if r['reid'] is None:
                    return
                if not os.path.exists(os.path.join(self.output_folder, out_name)):
                    return
                vest_full = np.concatenate([r['vest'], r['reid']]) \
                    if r['vest'] is not None else r['reid']
                n = np.linalg.norm(vest_full)
                store[out_name] = {
                    'vest': vest_full / max(n, 1e-9),
                    'boat': r['boat'] if r['boat'] is not None else np.zeros(18)}
                # предложение атлета сразу по факту появления попытки:
                # банки из текущего (в т.ч. только что подтверждённого) маппинга
                self.refresh_auto_assignments(store=store)

            for file_index, video_path in enumerate(self.selected_files):
                if not self.processing:
                    break
                base = os.path.splitext(os.path.basename(video_path))[0]
                self.log(f"[{file_index + 1}/{len(self.selected_files)}] "
                         f"Поиск попыток: {os.path.basename(video_path)} ...")
                tmp_dir = os.path.join(self.output_folder, f".attempts_tmp_{base}")
                try:
                    pstats = {}
                    results = attempts_v2.process_video(
                        video_path, tmp_dir, roi=roi_bbox,
                        should_continue=lambda: self.processing,
                        stats_out=pstats,
                        slowmo_factor=self._active_slowmo_factor,
                        min_pause=float(self.processing_params.get('min_pause_duration', 2.5)),
                        min_attempt_duration=float(self.processing_params.get('min_attempt_duration', 3.0)),
                        reid_embed_rate=float(self.processing_params.get('reid_embed_rate', 5)),
                        attempt_found_cb=lambda r, src=video_path: on_attempt_found({**r, 'source': src}),
                        log_cb=self.log,
                        progress_cb=lambda fi, tf, fi_=file_index, nf=len(self.selected_files):
                            self.on_processing_progress(fi_, nf, fi, tf))
                except Exception as e:
                    self.log(f"Ошибка обработки {base}: {e}", logging.ERROR)
                    continue
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if self.processing:
                    self._mark_processed(video_path, 'attempts')

                self._save_profile_store(store)
                if not results and self.processing:
                    pr = pstats.get('presence_ratio', 0.0)
                    self.log(f"  Попыток не найдено. Главный атлет присутствовал в ROI "
                             f"в {pr * 100:.0f}% времени. Если попытки в файле точно есть - "
                             f"проверьте ROI (атлет должен быть целиком внутри области) "
                             f"и пороги (мин. длительность попытки)", logging.WARNING)

                self._save_profile_store(store)

            # автоатрибуция новых попыток
            self._auto_assign_attempts(store, banks)
            self._log_processing_summary("Попытки (атрибуция)",
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            self.log(f"Ошибка режима попыток: {e} {traceback.format_exc()}", logging.ERROR)
        finally:
            self.processing = False
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_attempt_thumbnails)
            self.root.after(0, self.update_athlete_list)

    def _try_auto_assign(self, store, name, banks):
        """Уверенное авто-присвоение одной попытки (под lock вызывающего).
        True - присвоена (провизорная привязка сохранена сразу)."""
        sims = {a: self._attempt_athlete_sim(store[name], b)[0]
                for a, b in banks.items()}
        srt = sorted(sims.items(), key=lambda kv: -kv[1])
        best, best_sim = srt[0]
        second_sim = srt[1][1] if len(srt) >= 2 else 0.0
        if best_sim >= self.ATTEMPT_SIM_THRESHOLD and best_sim - second_sim >= self.ATTEMPT_SIM_MARGIN:
            self.attempt_assignments[name] = {'athlete': best,
                                              'sim': round(best_sim, 3),
                                              'second': round(second_sim, 3)}
            self._save_assignments()
            self.need_update_attempts = True  # плитки обновятся постепенно
            self.log(f"  Автоатрибуция: {name} → {best} (sim={best_sim:.2f}) "
                     f"- подтвердите на сетке превью")
            return True
        hint = ', '.join(f'{a}={s:.2f}' for a, s in srt[:2])
        self.log(f"  {name}: требует подтверждения ({hint})")
        return False

    def _auto_assign_attempts(self, store, banks):
        """Автоприсвоение новых попыток: уверенные - в provизорные привязки
        (assignments.yaml, до подтверждения пользователем), спорные - в лог.
        Банки строятся только из подтверждённого маппинга, поэтому ошибка
        авто-привязки не отравляет дальнейшую атрибуцию.
        Вызывается в конце прогона и после каждого подтверждения."""
        if not banks:
            return
        with self._assignments_lock:
            assigned_names = {os.path.basename(f)
                              for files in self.athlete_mapping.values() for f in files}
            fresh = [n for n in store
                     if n not in assigned_names and n not in self.attempt_assignments]
            n_auto = 0
            for name in fresh:
                if self._try_auto_assign(store, name, banks):
                    n_auto += 1
        if fresh:
            self.log(f"Автоатрибуция: предложено {n_auto} из {len(fresh)}")

    def refresh_auto_assignments(self, store=None):
        """Пересобирает банки (по подтверждённому маппингу) и немедленно
        перепредлагает авто-привязки - подтверждение учитывается сразу.
        store=None -> загрузить с диска; иначе используется живой store
        (вызов из on_attempt_found в рабочем потоке)."""
        try:
            if store is None:
                store = self._load_profile_store()
            banks = self._athlete_banks(store)
            if banks:
                self._auto_assign_attempts(store, banks)
        except Exception as e:
            self.log(f"Ошибка пересчёта автоатрибуции: {e}", logging.ERROR)

    def process_athlete_scan(self):
        """Стратегия 'Скан атлетов': тот же keyframe-сплиттер, но БЕЗ нарезки
        клипов - сегменты присутствия атлетов с мгновенной атрибуцией по
        банкам (локальный маппинг + глобальный реестр). Результаты - в
        segments.yaml + segment_profiles.npz, показываются на вкладке 'Скан'.
        Подтверждение сегмента пополняет реестр."""
        t0 = time.time()
        try:
            import attempts_v2
            roi_bbox = self._roi_bbox_percentages()
            self._prune_athlete_mapping()
            banks = self._athlete_banks(self._load_profile_store())
            if banks:
                self.log(f"Банки атлетов: {', '.join(banks.keys())}")
            else:
                self.log("Банков нет - сегменты будут 'не разобрано'; "
                         "подтвердите атлета на строке сегмента - банк пополнится")
            segs = list(self.athlete_segments)
            profiles = self._segment_profiles
            state = {'counter': 0, 'file': None}

            def on_segment_found(r):
                src = r['source']
                base = os.path.basename(src)
                if state['file'] != base:
                    # повторный скан файла заменяет его старые сегменты
                    state['file'] = base
                    state['counter'] = 0
                    segs[:] = [s for s in segs if s['file'] != base]
                    for k in [k for k in profiles if k.startswith(base + '#')]:
                        del profiles[k]
                key = f"{base}#{state['counter']}"
                state['counter'] += 1
                seg = {'key': key, 'file': base, 'source': src,
                       'start': round(float(r['start']), 2),
                       'end': round(float(r['end']), 2),
                       'thumb_t': round(float(r.get('thumb_t') or r['start']), 2),
                       'athlete': None, 'sim': None, 'confirmed': False}
                srt = []
                if r['reid'] is not None:
                    vest_full = np.concatenate([r['vest'], r['reid']]) \
                        if r['vest'] is not None else r['reid']
                    n = np.linalg.norm(vest_full)
                    prof = {'vest': vest_full / max(n, 1e-9),
                            'boat': r['boat'] if r['boat'] is not None else np.zeros(18)}
                    profiles[key] = prof
                    sims = {a: self._attempt_athlete_sim(prof, b)[0]
                            for a, b in banks.items()}
                    srt = sorted(sims.items(), key=lambda kv: -kv[1])
                    if srt:
                        best, s = srt[0]
                        second = srt[1][1] if len(srt) > 1 else 0.0
                        seg['sim'] = round(float(s), 3)
                        if s >= self.ATTEMPT_SIM_THRESHOLD and \
                                s - second >= self.ATTEMPT_SIM_MARGIN:
                            seg['athlete'] = best
                segs.append(seg)
                if seg['athlete']:
                    who = f"{seg['athlete']} {seg['sim'] * 100:.0f}%"
                elif srt:
                    who = f"{srt[0][0]} {seg['sim'] * 100:.0f}%?"
                else:
                    who = "не разобрано"
                self.log(f"  Сегмент: {who} — {seg['start']:.1f}–{seg['end']:.1f}с "
                         f"({base})")
                # живое обновление панели: сегменты появляются по мере скана
                self.athlete_segments = list(segs)
                self.need_update_segments = True

            for file_index, video_path in enumerate(self.selected_files):
                if not self.processing:
                    break
                base = os.path.splitext(os.path.basename(video_path))[0]
                self.log(f"[{file_index + 1}/{len(self.selected_files)}] "
                         f"Скан атлетов: {os.path.basename(video_path)} ...")
                tmp_dir = os.path.join(self.output_folder, f".scan_tmp_{base}")
                try:
                    attempts_v2.process_video(
                        video_path, tmp_dir, roi=roi_bbox,
                        should_continue=lambda: self.processing,
                        slowmo_factor=self._active_slowmo_factor,
                        min_pause=float(self.processing_params.get('min_pause_duration', 2.5)),
                        min_attempt_duration=float(self.processing_params.get('min_attempt_duration', 3.0)),
                        reid_embed_rate=float(self.processing_params.get('reid_embed_rate', 5)),
                        collect_tiles=False,
                        attempt_found_cb=lambda r, src=video_path: on_segment_found({**r, 'source': src}),
                        log_cb=self.log,
                        progress_cb=lambda fi, tf, fi_=file_index, nf=len(self.selected_files):
                            self.on_processing_progress(fi_, nf, fi, tf))
                except Exception as e:
                    self.log(f"Ошибка обработки {base}: {e}", logging.ERROR)
                    continue
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if self.processing:
                    self._mark_processed(video_path, 'athletes')

            self.athlete_segments = segs
            self._save_segments()
            self._save_segment_profiles()
            self.log(f"Скан атлетов завершён: {len(segs)} сегментов. "
                     f"Подтверждайте атлетов на вкладке 'Скан'")
            self._log_processing_summary("Скан атлетов",
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            self.log(f"Ошибка скана атлетов: {e} {traceback.format_exc()}", logging.ERROR)
        finally:
            self.processing = False
            self.need_update_segments = True
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self._raise_scan_tab)

    def confirm_segment(self, seg, athlete_name=None):
        """Подтверждение сегмента скана: атлет фиксируется, профиль сегмента
        пополняет глобальный реестр."""
        athlete = athlete_name or seg.get('athlete')
        if not athlete:
            return
        seg['athlete'] = athlete
        seg['confirmed'] = True
        prof = self._segment_profiles.get(seg.get('key'))
        if prof is not None:
            self._registry_add_sample(athlete, prof['vest'], prof['boat'])
            self.log(f"Сегмент подтверждён: {athlete} — {seg['start']:.1f}–"
                     f"{seg['end']:.1f}с ({seg['file']}) - банк пополнен")
        else:
            self.log(f"Сегмент подтверждён: {athlete} ({seg['file']})")
        self._save_segments()
        self.update_segments_panel()
        self.update_athlete_list()

    def remove_segment_assignment(self, seg):
        """Снимает привязку сегмента (в т.ч. подтверждение)."""
        seg['athlete'] = None
        seg['sim'] = None
        seg['confirmed'] = False
        self._save_segments()
        self.update_segments_panel()


    def on_candidate_found(self, attempt):
        """Callback сканирования (вызывается из рабочего потока): запоминаем кандидата.
        Ничего не режем - только микропревью и запись в состояние.
        Кандидаты, пересекающиеся с уже найденными, пропускаются."""
        with self._candidates_lock:
            for c in self.candidates:
                if c['file'] == attempt.source_video and \
                        max(c['start'], attempt.start) < min(c['end'], attempt.end):
                    return self.processing  # дубль области - пропускаем
            idx = len(self.candidates) + 1
            thumb_path = os.path.join(self.output_folder, f"cand_{idx:03d}.jpg")
            try:
                cv2.imwrite(thumb_path, attempt.best_frame)
            except Exception:
                thumb_path = None
            self.candidates.append({
                'file': attempt.source_video,
                'start': float(attempt.start),
                'end': float(attempt.end),
                'thumbnail': thumb_path,
                'selected': True,
                'status': 'pending',
            })
        self.need_update_attempts = True
        self.log(f"Кандидат #{idx}: {os.path.basename(attempt.source_video)} "
                 f"{attempt.start:.1f}s..{attempt.end:.1f}s")
        return self.processing

    def _candidate_key(self, cand):
        return (cand['file'], round(cand['start'], 2))

    def on_candidate_checkbox_change(self, cand, var):
        cand['selected'] = var.get()
        self.update_cut_button_state()
        self.save_candidates()

    def update_cut_button_state(self):
        """Обновляет кнопку нарезки выбранных кандидатов: количество и примерный объём выхлопа"""
        with self._candidates_lock:
            selected = [c for c in self.candidates
                        if c['status'] == 'pending' and c.get('selected')]
        n = len(selected)
        text = "✂️ Нарезать выбранное"
        if n > 0:
            text += f" ({n})"
            try:
                start_pad = self.processing_params['attempt_start_padding']
                end_pad = self.processing_params['attempt_end_padding']
                scan_interval = float(self.processing_params.get('scan_interval', 2.5))
                margin = scan_interval + 1.0  # как в _cut_candidates_thread
                total = sum(self._estimate_output_bytes(
                    c['file'], c['end'] - c['start'] + start_pad + end_pad + margin)
                    for c in selected)
                if total > 0:
                    text += f" ≈{self._fmt_size(total)}"
            except Exception:
                pass
            self.button_cut_selected.config(state=tk.NORMAL, text=text)
        else:
            self.button_cut_selected.config(state=tk.DISABLED, text=text)

    def cut_selected_candidates(self):
        """Нарезает выбранных кандидатов: точный анализ границ + вырезка.
        Может запускаться во время сканирования - идут параллельно."""
        with self._candidates_lock:
            selected = [c for c in self.candidates
                        if c['status'] == 'pending' and c.get('selected')]
        if not selected:
            messagebox.showwarning("Предупреждение", "Не выбрано ни одного кандидата")
            return
        if self.cut_running:
            return
        # Слоумо берём из чекбокса именно сейчас (а не из прошлого запуска
        # "Обработать"), чтобы нарезка кандидатов учитывала его.
        self._active_slowmo_factor = self._ui_slowmo_factor()
        self.cut_running = True
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        self.log(f"Нарезка {len(selected)} кандидатов, слоумо "
                 f"{'1/' + str(self._active_slowmo_factor) if self._active_slowmo_factor > 1 else 'нет'}")
        threading.Thread(target=self._cut_candidates_thread, args=(selected,), daemon=True).start()

    def _cut_candidates_thread(self, selected):
        """Точная стадия ручного режима: для каждого кандидата - полный анализ
        в узком окне вокруг него и вырезка найденных попыток."""
        t0 = time.time()
        try:
            slowmo = self._active_slowmo_factor
            model = load_yolo_model()
            start_pad = self.processing_params['attempt_start_padding']
            end_pad = self.processing_params['attempt_end_padding']
            scan_interval = float(self.processing_params.get('scan_interval', 2.5))
            margin = scan_interval + 1.0  # запас на неточность скана
            # окна уже нарезанных кандидатов (включая прошлые запуски) - чтобы не нарезать дубль
            with self._candidates_lock:
                cut_windows = [(c['file'], c['start'], c['end'])
                               for c in self.candidates if c['status'] == 'cut']
            cut_count = 0
            total = len(selected)
            stopped = False
            for i, cand in enumerate(selected):
                if not self.processing:
                    stopped = True
                    remaining = total - i
                    self.log(f"Нарезка остановлена: обработано {i} из {total}, "
                             f"осталось {remaining} кандидатов. Повторный запуск "
                             f"«Нарезать выбранное» продолжит с того же места.")
                    break
                left = total - i
                if any(f == cand['file'] and max(s, cand['start']) < min(e, cand['end'])
                       for f, s, e in cut_windows):
                    self.log(f"[{i + 1}/{total}, осталось {left - 1}] Кандидат "
                             f"{os.path.basename(cand['file'])} {cand['start']:.1f}s..{cand['end']:.1f}s "
                             f"пересекается с уже нарезанной попыткой - пропущен")
                    continue
                # влево ищем точное начало: margin по умолчанию (шаг скана + 1)
                # или явный 'precise_left_margin' (кандидат может опаздывать)
                left_margin = float(self.processing_params.get('precise_left_margin', 0)
                                    or 0) or margin
                win_start = max(0.0, cand['start'] - start_pad - left_margin)
                win_end = cand['end'] + end_pad + margin
                next_num = get_next_attempt_number(self.output_folder)
                self.log(f"[{i + 1}/{total}, осталось {left - 1}] Точный анализ "
                         f"{os.path.basename(cand['file'])} {cand['start']:.1f}s..{cand['end']:.1f}s")
                found = []

                def fine_callback(attempt, found=found):
                    found.append(attempt)
                    return self.on_attempt_file_created(attempt)

                try:
                    process_video([cand['file']], self.roi, fine_callback, next_num,
                                  params=self.processing_params,
                                  should_continue=lambda: self.processing,
                                  slowmo_factor=slowmo,
                                  model=model,
                                  restrict_range=(win_start, win_end))
                except Exception as e:
                    self.log(f"Ошибка точного анализа: {e} {traceback.format_exc()}", logging.ERROR)
                if found:
                    cand['status'] = 'cut'
                    cut_count += len(found)
                    cut_windows.append((cand['file'], cand['start'], cand['end']))
                else:
                    self.log("В окне кандидата попыток не найдено - пропущен")
                self.save_candidates()
            if not stopped:
                self.log(f"Нарезка завершена: попыток {cut_count} из {total} кандидатов")
                self._log_processing_summary("Нарезка выбранных кандидатов",
                                             [c['file'] for c in selected],
                                             t0, True)
        finally:
            self.cut_running = False
            self.processing = self.scan_running
            still_busy = self.processing
            self.need_update_attempts = True
            self.root.after(0, lambda: self._after_worker_ui(still_busy))

    def clear_pending_candidates(self):
        """Удаляет всех неподтверждённых (ненарезанных) кандидатов - для чистого
        пересканирования с нуля. Уже нарезанные не трогаются."""
        if self.cut_running:
            messagebox.showinfo("Кандидаты", "Дождитесь завершения нарезки")
            return
        with self._candidates_lock:
            pending = [c for c in self.candidates if c['status'] == 'pending']
            if not pending:
                messagebox.showinfo("Кандидаты", "Нет кандидатов для очистки")
                return
            if not messagebox.askyesno("Подтверждение",
                                       f"Удалить {len(pending)} кандидатов (ненарезанных)?"):
                return
            for cand in pending:
                thumb = cand.get('thumbnail')
                if thumb and os.path.exists(thumb):
                    try:
                        os.remove(thumb)
                    except OSError:
                        pass
            self.candidates = [c for c in self.candidates if c['status'] != 'pending']
        self.candidate_vars = {}
        self.save_candidates()
        self.update_cut_button_state()
        self.update_attempt_thumbnails()
        self._render_file_list()
        self.log(f"Кандидаты очищены (удалено {len(pending)})")

    def save_candidates(self):
        """Сохраняет кандидатов в candidates.yaml (потокобезопасно)"""
        with self._candidates_lock:
            if not self.candidates_file:
                return
            data = {'candidates': [
                {'file': c['file'], 'start': c['start'], 'end': c['end'],
                 'thumbnail': self.get_relative_path(c['thumbnail'])
                 if c.get('thumbnail') else None,
                 'selected': bool(c.get('selected')),
                 'status': c['status']}
                for c in self.candidates]}
        try:
            os.makedirs(os.path.dirname(self.candidates_file), exist_ok=True)
            with open(self.candidates_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)
        except Exception as e:
            self.log(f"Ошибка сохранения кандидатов: {e}")

    def load_candidates(self):
        """Загружает кандидатов из candidates.yaml"""
        self.candidates = []
        self.candidate_vars = {}
        if self.candidates_file and os.path.exists(self.candidates_file):
            try:
                with open(self.candidates_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                for c in data.get('candidates') or []:
                    thumb = c.get('thumbnail')
                    if thumb and not os.path.isabs(thumb):
                        thumb = os.path.join(self.output_folder, thumb)
                    if not thumb or not os.path.exists(thumb):
                        continue
                    self.candidates.append({
                        'file': self.get_absolute_path(c.get('file', '')),
                        'start': float(c.get('start', 0)),
                        'end': float(c.get('end', 0)),
                        'thumbnail': thumb,
                        'selected': bool(c.get('selected')),
                        'status': c.get('status', 'pending'),
                    })
            except Exception as e:
                self.log(f"Ошибка загрузки кандидатов: {e}")

    def on_processing_progress(self, file_index, file_count, frame_index, total_frames):
        """Прогресс обработки. Вызывается из рабочего потока, поэтому только считаем
        процент - сам прогрессбар обновляет periodic_update в главном потоке."""
        if file_count <= 0:
            return
        file_progress = frame_index / total_frames if total_frames > 0 else 0.0
        self._progress_percent = 100.0 * (file_index + min(file_progress, 1.0)) / file_count

    def setup_logger(self):
        """Настройка логгера"""
        self.logger = logging.getLogger('FreestyleParser')
        self.logger.setLevel(logging.INFO)

        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Хендлер для файла
        log_file = os.path.join(self.output_folder, "processing.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Хендлер для GUI
        self.gui_handler = GuiLogHandler(self)
        self.gui_handler.setFormatter(formatter)
        self.logger.addHandler(self.gui_handler)

    def get_relative_path(self, path):
        """Преобразует абсолютный путь в относительный относительно output_folder"""
        if not os.path.isabs(path):
            return path
        try:
            return os.path.relpath(path, self.output_folder)
        except ValueError:
            return path

    def get_absolute_path(self, path):
        """Преобразует относительный путь в абсолютный относительно output_folder"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.output_folder, path)

    def load_athlete_mapping(self):
        """Загружает маппинг атлетов из файла"""
        mapping_file = os.path.join(self.output_folder, "mapping.yaml")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    mapping = yaml.safe_load(f)
                    if mapping is None:  # Если файл пустой или содержит только комментарии
                        self.athlete_mapping = {}
                    else:
                        # Преобразуем относительные пути в абсолютные
                        self.athlete_mapping = {
                            name: list(set(attempts))  # Convert back to a list if needed
                            for name, attempts in mapping.items()
                        }
            except Exception as e:
                self.log(f"Ошибка при загрузке маппинга: {str(e)}")
                self.athlete_mapping = {}
        else:
            self.athlete_mapping = {}
        self._prune_athlete_mapping()

    def _prune_athlete_mapping(self):
        """Убирает из маппинга привязки к несуществующим файлам.
        Пустые атлеты не удаляются (атлет мог быть только что создан)."""
        pruned = []
        for athlete in list(self.athlete_mapping.keys()):
            kept = [name for name in self.athlete_mapping[athlete]
                    if os.path.exists(os.path.join(self.output_folder, name))]
            pruned += [f"{athlete}: {name}"
                       for name in self.athlete_mapping[athlete] if name not in kept]
            self.athlete_mapping[athlete] = kept
        if pruned:
            self.log(f"Убраны мёртвые привязки маппинга: {', '.join(pruned)}")
            self.save_athlete_mapping()
        return pruned

    def _load_assignments(self):
        """Загружает авто-привязки (провизорные, до подтверждения пользователем)."""
        f = os.path.join(self.output_folder, "assignments.yaml")
        if os.path.exists(f):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    self.attempt_assignments = yaml.safe_load(fh) or {}
            except Exception as e:
                self.log(f"Ошибка загрузки assignments: {e}")
                self.attempt_assignments = {}
        else:
            self.attempt_assignments = {}

    def _save_assignments(self):
        f = os.path.join(self.output_folder, "assignments.yaml")
        try:
            with open(f, 'w', encoding='utf-8') as fh:
                yaml.dump(self.attempt_assignments, fh,
                          default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.log(f"Ошибка сохранения assignments: {e}")

    def confirm_assignment(self, attempt, athlete_name=None):
        """Подтверждает привязку попытки (auto -> confirmed в mapping.yaml)."""
        a = self.attempt_assignments.get(attempt)
        athlete = athlete_name or (a or {}).get('athlete')
        if not athlete:
            return
        self.attempt_assignments.pop(attempt, None)
        if athlete not in self.athlete_mapping:
            self.athlete_mapping[athlete] = []
        if attempt not in self.athlete_mapping[athlete]:
            self.athlete_mapping[athlete].append(attempt)
        self._save_assignments()
        self.save_athlete_mapping()
        # профиль попытки - в глобальный реестр (банк живёт между папками)
        prof = self._load_profile_store().get(attempt)
        if prof is not None:
            self._registry_add_sample(athlete, prof['vest'], prof['boat'])
        self.log(f"Подтверждено: {attempt} → {athlete}")
        self._refresh_attempt_tile(attempt)
        self.update_athlete_list()
        self.refresh_auto_assignments()

    def remove_assignment(self, attempt):
        """Снимает любую привязку попытки (и авто, и подтверждённую)."""
        changed = self.attempt_assignments.pop(attempt, None) is not None
        for attempts in self.athlete_mapping.values():
            if attempt in attempts:
                attempts.remove(attempt)
                changed = True
        if changed:
            self._save_assignments()
            self.save_athlete_mapping()
            self.log(f"Привязка снята: {attempt}")
            self._refresh_attempt_tile(attempt)
            self.update_athlete_list()
            self.refresh_auto_assignments()

    def confirm_all_auto_assignments(self):
        """Подтверждает авто-привязки одним кликом. При активном фильтре -
        только видимые попытки, скрытые остаются провизорными."""
        if not self.attempt_assignments:
            messagebox.showinfo("Информация", "Нет авто-привязок для подтверждения.")
            return
        visible = {os.path.basename(f) for f in self.get_filtered_attempts()}
        n = 0
        touched = []
        for attempt, a in list(self.attempt_assignments.items()):
            if attempt not in visible:
                continue
            athlete = a.get('athlete')
            if not athlete:
                continue
            self.attempt_assignments.pop(attempt, None)
            if athlete not in self.athlete_mapping:
                self.athlete_mapping[athlete] = []
            if attempt not in self.athlete_mapping[athlete]:
                self.athlete_mapping[athlete].append(attempt)
            touched.append(attempt)
            n += 1
        self._save_assignments()
        self.save_athlete_mapping()
        self.log(f"Подтверждено авто-привязок: {n}")
        for attempt in touched:
            self._refresh_attempt_tile(attempt)
        self.update_athlete_list()
        self.refresh_auto_assignments()

    def save_athlete_mapping(self):
        mapping_file = os.path.join(self.output_folder, "mapping.yaml")
        try:
            # Преобразуем абсолютные пути в относительные
            mapping = {
                name: list(set([self.get_relative_path(path) for path in attempts]))
                for name, attempts in self.athlete_mapping.items()
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                yaml.dump(mapping, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.log(f"Ошибка при сохранении маппинга: {str(e)}")

    def save_ratings(self):
        """Сохраняет рейтинги попыток в файл"""
        ratings_file = os.path.join(self.output_folder, "ratings.yaml")
        try:
            with open(ratings_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.attempt_ratings, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.log(f"Ошибка при сохранении рейтингов: {str(e)}")

    def load_ratings(self):
        """Загружает рейтинги попыток из файла"""
        ratings_file = os.path.join(self.output_folder, "ratings.yaml")
        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r', encoding='utf-8') as f:
                    self.attempt_ratings = yaml.safe_load(f) or {}
            except Exception as e:
                self.log(f"Ошибка при загрузке рейтингов: {str(e)}")
                self.attempt_ratings = {}
        else:
            self.attempt_ratings = {}

    def add_athlete(self):
        """Добавляет нового атлета (с подсказками из банков)"""
        name = self._ask_athlete(self.root)
        if name:
            # Создаем виджет атлета
            athlete_widget = AthleteWidget(self.athletes_container, name, self)
            athlete_widget.pack(fill=tk.X, pady=2)
            self.athlete_widgets.append(athlete_widget)

            # Обновляем маппинг
            if name not in self.athlete_mapping:
                self.athlete_mapping[name] = []
            self.save_athlete_mapping()

            self.log(f"Добавлен новый атлет: {name}")

    def assign_attempt_to_athlete(self, filename, athlete_name):
        """Привязывает попытку к атлету (ручное действие = подтверждённая)."""
        # Ручная привязка снимает провизорную
        if self.attempt_assignments.pop(filename, None) is not None:
            self._save_assignments()

        # Удаляем попытку из всех атлетов
        for attempts in self.athlete_mapping.values():
            if filename in attempts:
                attempts.remove(filename)

        # Если указан атлет, добавляем попытку к нему
        if athlete_name:
            if athlete_name not in self.athlete_mapping:
                self.athlete_mapping[athlete_name] = []
            if filename not in self.athlete_mapping[athlete_name]:
                self.athlete_mapping[athlete_name].append(filename)

        # Сохраняем изменения
        self.save_athlete_mapping()

        # Обновляем отображение
        self._refresh_attempt_tile(filename)
        self.update_athlete_list()
        self.refresh_auto_assignments()

    def update_athlete_list(self):
        """Обновляет список атлетов"""
        # Очищаем контейнер
        for widget in self.athlete_widgets:
            widget.destroy()
        self.athlete_widgets.clear()

        # Добавляем виджеты атлетов
        for name in sorted(self.athlete_mapping.keys()):
            athlete_widget = AthleteWidget(self.athletes_container, name, self)
            athlete_widget.pack(fill=tk.X, pady=2)
            self.athlete_widgets.append(athlete_widget)

            # Устанавливаем состояние выбора
            athlete_widget.set_selected(name == self.filter_var.get())

            # Привязываем обработчик клика
            athlete_widget.button.configure(command=lambda n=name: self.set_filter(n))

    def setup_ui(self):
        # === Основной контейнер с вертикальной ориентацией ===
        self.main_frame = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === Верхний контейнер с двумя частями ===
        self.top_frame = tk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.main_frame.add(self.top_frame)

        # === ЛЕВЫЙ БЛОК ===
        self.left_frame = tk.Frame(self.top_frame, width=460)
        self.top_frame.add(self.left_frame, width=460, minsize=425)

        # --- Выбор файлов ---
        self.frame_files = tk.LabelFrame(self.left_frame, text="Файлы")
        self.frame_files.pack(pady=10, fill=tk.X)

        files_inner = tk.Frame(self.frame_files)
        files_inner.pack(fill=tk.X, padx=5, pady=5)

        # Левая часть — входные файлы (фиксированная ширина)
        files_left = tk.Frame(files_inner)
        files_left.pack(side=tk.LEFT)

        self.label_files = tk.Label(files_left, text="Входные файлы:")
        self.label_files.pack(anchor=tk.W)

        self.listbox_files = tk.Listbox(files_left, height=5, width=22)
        self.listbox_files.pack(fill=tk.X, padx=5)
        self.listbox_files.bind('<<ListboxSelect>>', self.on_file_select)

        self.button_select_files = tk.Button(files_left, text="Выбрать файлы", command=self.select_files)
        self.button_select_files.pack(fill=tk.X, padx=5, pady=5)

        # Правая часть — выходная папка
        files_right = tk.Frame(files_inner)
        files_right.pack(side=tk.LEFT, padx=(10, 0))

        self.label_output = tk.Label(files_right, text="Выходная папка:")
        self.label_output.pack(anchor=tk.W)

        # Имя рабочей папки (только чтение) + кнопка открытия в той же строке
        entry_row = tk.Frame(files_right)
        entry_row.pack(fill=tk.X, padx=5, pady=2)

        self.entry_output = tk.Entry(entry_row, width=11, state='readonly', justify='center')
        self.entry_output.pack(side=tk.LEFT)
        self.entry_output.config(state='normal')
        self.entry_output.insert(0, os.path.basename(self.output_folder))
        self.entry_output.config(state='readonly')

        self.button_open_output = tk.Button(entry_row, text="Открыть", command=self.open_output_folder)
        self.button_open_output.pack(side=tk.LEFT, padx=(5, 0))

        output_btn_inner = tk.Frame(files_right)
        output_btn_inner.pack(fill=tk.X, padx=5, pady=5)

        self.button_day_folder = tk.Button(output_btn_inner, text="📅",
                                           command=self.choose_day_folder)
        self.button_day_folder.pack(side=tk.LEFT, padx=2)

        self.button_change_output = tk.Button(output_btn_inner, text="Изменить", command=self.change_output_folder)
        self.button_change_output.pack(side=tk.LEFT, padx=2)

        # Индикатор состояния рабочей папки
        self.folder_status_label = tk.Label(files_right, text="", anchor=tk.W)
        self.folder_status_label.pack(fill=tk.X, padx=5)

        # --- ROI Canvas ---
        self.frame_roi = tk.LabelFrame(self.left_frame, text="Область интереса")
        self.frame_roi.pack(pady=10, fill=tk.X)

        # Кнопки для переключения режимов ROI
        self.frame_roi_controls = tk.Frame(self.frame_roi)
        self.frame_roi_controls.pack(pady=5)
        
        self.button_rectangle_mode = tk.Button(self.frame_roi_controls, text="Прямоугольник", 
                                              command=lambda: self.switch_roi_mode("rectangle"))
        self.button_rectangle_mode.pack(side=tk.LEFT, padx=5)
        
        self.button_polygon_mode = tk.Button(self.frame_roi_controls, text="Многоугольник", 
                                            command=lambda: self.switch_roi_mode("polygon"))
        self.button_polygon_mode.pack(side=tk.LEFT, padx=5)
        
        self.button_clear_roi = tk.Button(self.frame_roi_controls, text="Очистить", 
                                         command=self.clear_roi)
        self.button_clear_roi.pack(side=tk.LEFT, padx=5)

        self.label_roi = tk.Label(self.frame_roi, text="Кликните по точкам многоугольника. Двойной клик завершает:")
        self.label_roi.pack()

        self.canvas = tk.Canvas(self.frame_roi, height=CANVAS_HEIGHT, bg="black")
        self.canvas.pack(pady=5, fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)  # Двойной клик завершает многоугольник

        # --- Параметры обработки ---
        self.frame_output = tk.LabelFrame(self.left_frame, text="Настройки обработки")
        self.frame_output.pack(pady=10, fill=tk.X)

        param_defs = [
            ('min_pause_duration', 'Пауза между попытками'),
            ('min_attempt_duration', 'Мин. длит. попытки'),
            ('attempt_start_padding', 'Запас к началу'),
            ('attempt_end_padding', 'Запас к концу'),
            ('min_detection_strength', 'Мин. сила детекции (0-1)'),
            ('scan_interval', 'Шаг скана, с'),
            ('scan_threshold', 'Порог скана (0-1)'),
            ('reid_embed_rate', 'OSNet сэмплов/с в попытке'),
            ('precise_left_margin', 'Поиск начала влево, с'),
        ]

        self.param_vars = {}
        for i, (key, label_text) in enumerate(param_defs):
            col = i % 2
            if col == 0:
                row_frame = tk.Frame(self.frame_output)
                row_frame.pack(fill=tk.X, pady=2, padx=5)
            label = tk.Label(row_frame, text=label_text, anchor=tk.W)
            label.pack(side=tk.LEFT, padx=(0, 2))
            var = tk.StringVar(value=str(self.processing_params[key]))
            entry = tk.Entry(row_frame, textvariable=var, width=6)
            entry.pack(side=tk.LEFT, padx=(0, 15))
            self.param_vars[key] = var

        # --- Слоумо: чекбокс + коэффициент (в той же сетке параметров) ---
        i = len(param_defs)
        if i % 2 == 0:
            row_frame = tk.Frame(self.frame_output)
            row_frame.pack(fill=tk.X, pady=2, padx=5)
        self.slowmo_var = tk.BooleanVar(value=False)
        self.check_slowmo = tk.Checkbutton(
            row_frame, text="Слоумо", variable=self.slowmo_var,
            command=self.on_slowmo_toggle)
        self.check_slowmo.pack(side=tk.LEFT)
        self.slowmo_combo_var = tk.StringVar(value='1/4')
        self.slowmo_combo = ttk.Combobox(
            row_frame, textvariable=self.slowmo_combo_var,
            values=('1/4', '1/8'), width=4, state='disabled')
        self.slowmo_combo.pack(side=tk.LEFT, padx=(2, 0))

        # --- Стратегия обработки: 2 чекбокса = 4 комбинации ---
        # [ ] Сразу сохранять  [ ] Определять атлетов
        # ☑☐ - резать всё найденное; ☐☐ - превью-кандидаты для отбора;
        # ☐☑ - сегменты+атлеты в оригиналах (скан атлетов); ☑☑ - резать всё
        # с привязкой к атлетам
        strategy_frame = tk.Frame(self.frame_output)
        strategy_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        tk.Label(strategy_frame, text="Стратегия:").pack(side=tk.LEFT, padx=(0, 5))
        self.save_now_var = tk.BooleanVar(value=True)
        self.athletes_var = tk.BooleanVar(value=False)
        tk.Checkbutton(strategy_frame, text="Сразу сохранять",
                       variable=self.save_now_var,
                       command=self.on_strategy_change).pack(side=tk.LEFT)
        tk.Checkbutton(strategy_frame, text="Определять атлетов",
                       variable=self.athletes_var,
                       command=self.on_strategy_change).pack(side=tk.LEFT, padx=(5, 0))

        # --- Управление процессом ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frame_output, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        control_inner = tk.Frame(self.frame_output)
        control_inner.pack(fill=tk.X)

        self.button_process = tk.Button(control_inner, text="Обработать", command=self.start_processing)
        self.button_process.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.button_stop = tk.Button(control_inner, text="Остановить", state=tk.DISABLED, command=self.stop_processing)
        self.button_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # === ПРАВЫЙ БЛОК ===
        self.right_frame = tk.PanedWindow(self.top_frame, orient=tk.HORIZONTAL)
        self.top_frame.add(self.right_frame)

        # --- Попытки ---
        self.frame_attempts = tk.LabelFrame(self.right_frame, text="Попытки")
        # stretch=always: вся лишняя ширина окна - сетке попыток
        self.right_frame.add(self.frame_attempts, width=950, stretch="always")

        # Область действий
        self.actions_frame = tk.Frame(self.frame_attempts)
        self.actions_frame.pack(fill=tk.X, padx=5, pady=5)

        # Счетчик выбранных попыток
        self.selected_count_label = tk.Label(self.actions_frame, text="Выбрано: 0")
        self.selected_count_label.pack(side=tk.LEFT, padx=5)

        # Экспорт выбранных попыток (видео + превью) в стороннюю папку
        self.export_button = tk.Button(self.actions_frame, text="Экспорт в...",
                                       state=tk.DISABLED,
                                       command=self.export_selected_attempts)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Фильтры по меткам
        self.rating_filters_frame = tk.Frame(self.actions_frame)
        self.rating_filters_frame.pack(side=tk.LEFT, padx=20)

        # Кнопка фильтра "палец вверх"
        self.filter_up_button = tk.Button(self.rating_filters_frame, text="👍", 
                                        command=lambda: self.toggle_rating_filter("up"),
                                        font=('Arial', 11, 'normal'), bd=2, width=3, height=1)
        self.filter_up_button.pack(side=tk.LEFT, padx=2)

        # Кнопка фильтра "палец вниз"
        self.filter_down_button = tk.Button(self.rating_filters_frame, text="👎", 
                                          command=lambda: self.toggle_rating_filter("down"),
                                          font=('Arial', 11, 'normal'), bd=2, width=3, height=1)
        self.filter_down_button.pack(side=tk.LEFT, padx=2)

        # Кнопка "Выбрать все"
        self.select_all_button = tk.Button(self.actions_frame, text="Выбрать все", command=self.select_all_attempts)
        self.select_all_button.pack(side=tk.RIGHT, padx=5)

        # Кнопка "Снять выделение"
        self.deselect_all_button = tk.Button(self.actions_frame, text="Снять выделение", command=self.deselect_all_attempts)
        self.deselect_all_button.pack(side=tk.RIGHT, padx=5)

        # Кнопка удаления
        self.delete_button = tk.Button(self.actions_frame, text="🗑️ Удалить", command=self.delete_selected_attempts)
        self.delete_button.pack(side=tk.RIGHT, padx=5)

        # Кнопка нарезки выбранных кандидатов (ручной режим).
        # Ярко выделена, чтобы читалась как кнопка, а не надпись.
        self.button_cut_selected = tk.Button(
            self.actions_frame, text="✂️ Нарезать выбранное",
            command=self.cut_selected_candidates, state=tk.DISABLED,
            relief=tk.RAISED, bd=2, bg="#dff0d8", activebackground="#c8e6c9",
            font=('Arial', 10, 'bold'), padx=8, pady=2)
        self.button_cut_selected.pack(side=tk.RIGHT, padx=5)

        # Создаем canvas и scrollbar для попыток
        self.attempts_canvas = tk.Canvas(self.frame_attempts)
        self.attempts_scrollbar = ttk.Scrollbar(self.frame_attempts, orient="vertical", command=self.attempts_canvas.yview)
        self.attempts_scrollable_frame = tk.Frame(self.attempts_canvas)

        self.attempts_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.attempts_canvas.configure(scrollregion=self.attempts_canvas.bbox("all"))
        )

        self.attempts_canvas.create_window((0, 0), window=self.attempts_scrollable_frame, anchor="nw")
        self.attempts_canvas.configure(yscrollcommand=self.attempts_scrollbar.set)

        # Размещаем canvas и scrollbar
        self.attempts_canvas.pack(side="left", fill="both", expand=True)
        self.attempts_scrollbar.pack(side="right", fill="y")

        # --- Атлеты ---
        # === Атлеты + Скан атлетов: одна панель, вкладки нарисованы вручную
        # (ttk.Notebook на macOS aqua рисует толстую тематическую рамку
        # вокруг области табов с внутренними отступами - стилем не убираются) ===
        self.frame_athletes = tk.LabelFrame(self.right_frame, text="Атлеты / Скан")
        # stretch=never: фиксированная ширина, без пустот по краям
        self.right_frame.add(self.frame_athletes, width=235, stretch="never",
                             minsize=225)
        tabbar = tk.Frame(self.frame_athletes, bd=0, highlightthickness=0)
        tabbar.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._athlete_tab_buttons = {}
        self._athlete_tab_frames = {}
        content = tk.Frame(self.frame_athletes, bd=0, highlightthickness=0)
        content.pack(fill=tk.BOTH, expand=True)
        for key, title in (('athletes', '👤 Атлеты'), ('scan', '🔍 Скан')):
            btn = tk.Label(tabbar, text=title, cursor='hand2',
                           font=('Arial', 10), padx=6, pady=2)
            btn.pack(side=tk.LEFT)
            btn.bind('<Button-1>',
                     lambda e, k=key: self._select_athlete_tab(k))
            self._athlete_tab_buttons[key] = btn
            self._athlete_tab_frames[key] = tk.Frame(content,
                                                     bd=0, highlightthickness=0)
            self._athlete_tab_frames[key].place(relx=0, rely=0,
                                                relwidth=1, relheight=1)
        self.tab_segments = self._athlete_tab_frames['scan']
        self._select_athlete_tab('athletes')

        # Кнопки фильтров
        tab_athletes = self._athlete_tab_frames['athletes']
        self.filter_buttons_frame = tk.Frame(tab_athletes)
        self.filter_buttons_frame.pack(fill=tk.X, padx=2, pady=3)

        self.filter_all_button = tk.Button(self.filter_buttons_frame, text="Все",
                                           command=lambda: self.set_filter("Все"))
        self.filter_all_button.pack(fill=tk.X, pady=2)

        self.filter_unknown_button = tk.Button(self.filter_buttons_frame, text="Не разобрано",
                                               command=lambda: self.set_filter("Неизвестно"))
        self.filter_unknown_button.pack(fill=tk.X, pady=2)

        self.filter_confirm_button = tk.Button(self.filter_buttons_frame, text="На подтверждение",
                                               command=lambda: self.set_filter("На подтверждение"))
        self.filter_confirm_button.pack(fill=tk.X, pady=2)

        self.confirm_auto_button = tk.Button(self.filter_buttons_frame,
                                             text="Подтвердить все авто",
                                             command=self.confirm_all_auto_assignments)
        self.confirm_auto_button.pack(fill=tk.X, pady=2)

        self.regenerate_thumbs_button = tk.Button(self.filter_buttons_frame,
                                                  text="Обновить превью",
                                                  command=self.start_thumbnail_regeneration)
        self.regenerate_thumbs_button.pack(fill=tk.X, pady=2)

        self.bank_detect_button = tk.Button(self.filter_buttons_frame,
                                            text="Детекция по банкам",
                                            command=self.start_bank_detection)
        self.bank_detect_button.pack(fill=tk.X, pady=2)

        # Контейнер для списка атлетов: скроллируемый (много атлетов)
        self.athletes_canvas = tk.Canvas(tab_athletes, highlightthickness=0)
        ath_sb = ttk.Scrollbar(tab_athletes, orient="vertical",
                               command=self.athletes_canvas.yview)
        self.athletes_container = tk.Frame(self.athletes_canvas)
        self.athletes_container.bind(
            "<Configure>",
            lambda e: self.athletes_canvas.configure(
                scrollregion=self.athletes_canvas.bbox("all")))
        self.athletes_canvas.create_window((0, 0), window=self.athletes_container,
                                           anchor="nw", width=210)
        self.athletes_canvas.configure(yscrollcommand=ath_sb.set)
        # порядок паковки важен: кнопка к низу и скроллбар вправо - ДО
        # канваса, иначе канвас (fill=both) забирает всю высоту/ширину
        self.button_add_athlete = tk.Button(tab_athletes, text="+ Новый атлет", command=self.add_athlete)
        self.button_add_athlete.pack(side=tk.BOTTOM, pady=3, padx=2, fill=tk.X)
        ath_sb.pack(side="right", fill="y")
        self.athletes_canvas.pack(side="left", fill="both", expand=True,
                                  padx=(2, 0), pady=3)
        self._bind_canvas_wheel(self.athletes_canvas)

        # === Вкладка Скан: статус банка + сегменты ===
        tab_segments = self._athlete_tab_frames['scan']
        self.bank_status_var = tk.StringVar(value="")
        tk.Label(tab_segments, textvariable=self.bank_status_var, fg='#555',
                 font=('Arial', 8)).pack(fill=tk.X, padx=3, pady=(3, 0))
        self.segments_canvas = tk.Canvas(tab_segments, highlightthickness=0)
        seg_sb = ttk.Scrollbar(tab_segments, orient="vertical",
                               command=self.segments_canvas.yview)
        self.segments_inner = tk.Frame(self.segments_canvas)
        self.segments_inner.bind(
            "<Configure>",
            lambda e: self.segments_canvas.configure(
                scrollregion=self.segments_canvas.bbox("all")))
        self.segments_canvas.create_window((0, 0), window=self.segments_inner,
                                           anchor="nw")
        self.segments_canvas.configure(yscrollcommand=seg_sb.set)
        self.segments_canvas.pack(side="left", fill="both", expand=True,
                                  padx=(2, 0), pady=3)
        seg_sb.pack(side="right", fill="y")
        self._bind_canvas_wheel(self.segments_canvas)
        self.update_segments_panel()

        # Инициализируем список атлетов
        self.update_athlete_list()

        # === ЛОГИ ===
        # Логи: сворачиваемая панель, по умолчанию скрыта.
        # Кнопка-язычок живёт ПОВЕРХ основного окна (в скрытой панели
        # она была бы не видна), в левом нижнем углу.
        self.logs_btn = tk.Label(self.main_frame, text='▸ Логи',
                                 cursor='hand2', font=('Arial', 10, 'bold'),
                                 padx=4, bg='#e8e8e8')
        self.logs_btn.place(relx=0.0, rely=1.0, x=4, y=-4, anchor='sw')
        self.logs_btn.lift()
        self.logs_btn.bind('<Button-1>', lambda e: self._toggle_logs())
        self.frame_logs_wrap = tk.Frame(self.main_frame)
        self.main_frame.add(self.frame_logs_wrap)
        self.frame_logs = tk.LabelFrame(self.frame_logs_wrap, text="Логи")
        self.frame_logs.pack(fill=tk.BOTH, expand=True)
        # панель скрыта по умолчанию (прячем саму pane PanedWindow)
        self._logs_visible = False
        self.main_frame.paneconfigure(self.frame_logs_wrap, hide=True)

        # Создаем текстовое поле для логов с прокруткой
        self.log_text = tk.Text(self.frame_logs, height=10, wrap=tk.WORD)
        self.log_scrollbar = ttk.Scrollbar(self.frame_logs, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        # Размещаем текстовое поле и скроллбар
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scrollbar.pack(side="right", fill="y")

        # Делаем текстовое поле только для чтения
        self.log_text.configure(state='disabled')

        # Схлопываемые панели левого блока (кнопка '-/+' в шапке)
        for f in (self.frame_files, self.frame_roi, self.frame_output):
            self._make_collapsible(f)

        # Справка по настройкам обработки: '?' слева от кнопки схлопывания
        self.help_btn = tk.Label(self.frame_output, text='?', cursor='hand2',
                                 font=('Arial', 10, 'bold'), padx=5, pady=0,
                                 bd=1, relief='solid', fg='#0b57d0',
                                 bg='#f0f0f0')
        self.help_btn.place(relx=1.0, x=-36, y=2, anchor='ne')
        self.help_btn.bind('<Button-1>', lambda e: self._open_help())

        # Панель 'Атлеты / Скан': скрываемая целиком (кнопка-язычок внизу
        # справа; при скрытии панель попыток расширяется)
        self._ath_pane_visible = True
        self.ath_toggle = tk.Label(self.right_frame, text='⇥', cursor='hand2',
                                   font=('Arial', 9, 'bold'), padx=3,
                                   bg='#e8e8e8')
        self.ath_toggle.place(relx=1.0, rely=1.0, x=-26, y=-4, anchor='se')
        self.ath_toggle.bind('<Button-1>', lambda e: self._toggle_athletes())

    def _make_collapsible(self, frame):
        """Схлопывание панели: узкая шапка-строка с маленькой кнопкой
        '-/+' (тонкий бордер) в правом углу; шапка видна и в свернутом
        виде. На Tk 9 pack_forget не пересчитывает высоту контейнера,
        поэтому при сворачивании фиксируем высоту."""
        frame._collapse_info = {}
        header = tk.Frame(frame)
        kids = frame.winfo_children()
        if kids:
            header.pack(side=tk.TOP, fill=tk.X, before=kids[0])
        else:
            header.pack(side=tk.TOP, fill=tk.X)
        btn = tk.Label(header, text='−', cursor='hand2',
                       font=('Arial', 10, 'bold'), padx=5, pady=0,
                       bd=1, relief='solid', bg='#f0f0f0')
        btn.pack(side=tk.RIGHT, padx=3, pady=1)

        def toggle(_=None):
            frame._open = not getattr(frame, '_open', True)
            btn.config(text='−' if frame._open else '+')
            for w in frame.winfo_children():
                if w is header:
                    continue
                if frame._open:
                    info = frame._collapse_info.get(w)
                    if info:
                        try:
                            w.pack(**info)
                        except tk.TclError:
                            pass
                else:
                    if w not in frame._collapse_info:
                        try:
                            frame._collapse_info[w] = w.pack_info()
                        except tk.TclError:
                            pass
                    w.pack_forget()
            if frame._open:
                frame.pack_propagate(True)
                frame.configure(height=0)
            else:
                frame.pack_propagate(False)
                frame.configure(height=30)
                # фоновые апдейты могут перепаковать детей - повторно скрываем
                frame.winfo_toplevel().after_idle(rehide)

        def rehide():
            if not getattr(frame, '_open', True):
                for w in frame.winfo_children():
                    if w is not header and w.winfo_ismapped():
                        w.pack_forget()

        btn.bind('<Button-1>', toggle)
        frame._collapse_btn = btn
        frame._collapse_toggle = toggle
        return toggle

    def _open_help(self):
        """Открывает HTML-справку в браузере (docs/help.html; в PyInstaller -
        из распакованного бандла)"""
        import webbrowser
        base = getattr(sys, '_MEIPASS', None) or os.path.dirname(
            os.path.abspath(__file__))
        path = os.path.join(base, 'docs', 'help.html')
        if not os.path.exists(path):
            messagebox.showinfo("Справка", f"Файл справки не найден: {path}")
            return
        webbrowser.open('file://' + path)

    def _toggle_athletes(self):
        """Показать/скрыть правую панель 'Атлеты / Скан' (PanedWindow pane)"""
        if getattr(self, '_ath_pane_visible', True):
            self.right_frame.forget(self.frame_athletes)
            self.ath_toggle.config(text='⇤')
            self._ath_pane_visible = False
        else:
            # Tk 9: у panedwindow нет insert - панель атлетов всегда была
            # последней, поэтому просто add в конец
            self.right_frame.add(self.frame_athletes, width=235,
                                 stretch="never", minsize=225)
            self.ath_toggle.config(text='⇥')
            self._ath_pane_visible = True

    def _bind_canvas_wheel(self, canvas):
        """Колесо мыши скроллит канвас, пока курсор над ним (macOS)"""
        def wheel(e):
            try:
                canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units')
            except tk.TclError:
                pass
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', wheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

    def _toggle_logs(self):
        """Показать/скрыть панель логов (скрываем pane PanedWindow)"""
        if getattr(self, '_logs_visible', False):
            self.main_frame.paneconfigure(self.frame_logs_wrap, hide=True)
            self.logs_btn.config(text='▸ Логи')
            self._logs_visible = False
        else:
            # Tk 9 не перераспределяет pane по reqsize - пересоздаём pane
            # с явной высотой
            self.main_frame.forget(self.frame_logs_wrap)
            self.main_frame.add(self.frame_logs_wrap, height=180, minsize=100,
                                stretch='never')
            self.logs_btn.config(text='▾ Логи')
            self._logs_visible = True
        self.logs_btn.lift()

    def set_filter(self, filter_name):
        """Устанавливает фильтр и обновляет отображение"""
        print(f"DEBUG: set_filter called with {filter_name}")  # Отладочное сообщение
        self.filter_var.set(filter_name)

        # Обновляем визуальное состояние всех атлетов
        for widget in self.athlete_widgets:
            widget.set_selected(widget.name == filter_name)

        # Обновляем отображение попыток
        self.update_attempt_thumbnails()

        # Обновляем список атлетов
        self.update_athlete_list()

        self.log(f"Установлен фильтр: {filter_name}")

    def update_attempt_thumbnails(self):
        """Обновляет превью попыток"""
        # сохраняем позицию скролла, чтобы перерисовка её не сбрасывала
        scroll_pos = self.attempts_canvas.yview()
        # Очищаем текущие превью
        for widget in self.attempts_scrollable_frame.winfo_children():
            widget.destroy()
        self._attempt_labels = {}
        self._attempt_attr_frames = {}

        # Получаем отфильтрованные попытки
        filtered_files = self.get_filtered_attempts()
        attempts = [os.path.basename(f) for f in filtered_files]

        # Очищаем выбранные попытки, которые больше не существуют
        self.selected_attempts = {attempt for attempt in self.selected_attempts if attempt in attempts}

        # Очищаем словарь чекбоксов
        self.attempt_checkboxes.clear()

        # Обновляем счетчик
        self._update_selected_count()

        # Сортируем попытки по имени
        attempts.sort()

        # Создаем сетку для превью
        row = 0
        col = 0
        max_cols = 4  # Фиксированное количество колонок

        # вероятности атрибуции - один расчёт на всё обновление сетки
        sims_cache = {}
        try:
            store = self._load_profile_store()
            banks = self._athlete_banks(store)
            for a in attempts:
                if a in store and banks:
                    sims_cache[a] = sorted(
                        ((ath, self._attempt_athlete_sim(store[a], b)[0])
                         for ath, b in banks.items()), key=lambda kv: -kv[1])
        except Exception:
            sims_cache = {}

        for attempt in attempts:
            # Создаем фрейм для превью
            frame = tk.Frame(self.attempts_scrollable_frame)
            frame.grid(row=row, column=col, padx=5, pady=5)

            # Создаем превью
            thumbnail_path = os.path.join(self.output_folder, f"{os.path.splitext(attempt)[0]}.jpg")
            if not os.path.exists(thumbnail_path):
                self.generate_thumbnail(attempt, thumbnail_path, self.roi)

            if os.path.exists(thumbnail_path):
                try:
                    # Загружаем изображение
                    image = Image.open(thumbnail_path)
                    # Изменяем размер
                    image = image.resize((200, 150), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)

                    # Создаем метку с изображением
                    label = tk.Label(frame, image=photo)
                    label.image = photo  # Сохраняем ссылку
                    label.pack()

                    # Создаем фрейм для имени файла и чекбокса
                    name_frame = tk.Frame(frame)
                    name_frame.pack()

                    # Создаем чекбокс
                    var = tk.BooleanVar(value=attempt in self.selected_attempts)
                    checkbox = tk.Checkbutton(name_frame, variable=var, 
                                            command=lambda a=attempt, v=var: self.on_attempt_checkbox_change(a, v))
                    checkbox.pack(side=tk.LEFT, padx=(0, 5))
                    
                    # Сохраняем ссылку на чекбокс
                    self.attempt_checkboxes[attempt] = var

                    # Добавляем имя файла
                    name_label = tk.Label(name_frame, text=os.path.basename(attempt))
                    name_label.pack(side=tk.LEFT)

                    # Добавляем кнопки рейтинга
                    rating_frame = tk.Frame(name_frame)
                    rating_frame.pack(side=tk.RIGHT, padx=(10, 0))

                    # Инициализируем рейтинг для попытки, если его нет
                    if attempt not in self.attempt_ratings:
                        self.attempt_ratings[attempt] = {'up': False, 'down': False}

                    # Кнопка "палец вверх"
                    up_button = tk.Button(rating_frame, text="👍", width=3, height=1,
                                       command=lambda a=attempt: self.toggle_attempt_rating(a, "up"),
                                       font=('Arial', 9, 'normal'), bd=1)
                    up_button.pack(side=tk.LEFT, padx=1)
                    
                    # Кнопка "палец вниз"
                    down_button = tk.Button(rating_frame, text="👎", width=3, height=1,
                                         command=lambda a=attempt: self.toggle_attempt_rating(a, "down"),
                                         font=('Arial', 9, 'normal'), bd=1)
                    down_button.pack(side=tk.LEFT, padx=1)

                    # Обновляем стиль кнопок в зависимости от текущего состояния
                    self.update_rating_button_style(up_button, down_button, self.attempt_ratings[attempt])

                    # Добавляем обработчики событий
                    label.bind('<Button-1>', lambda e, a=attempt: self.on_attempt_drag_start(e, a))
                    label.bind('<Double-Button-1>', lambda e, a=attempt: self.on_attempt_double_click(e, a))

                    # Статус атрибуции: подтверждено / авто / нет + быстрые действия
                    self._attempt_labels[attempt] = label
                    assigned_athlete = next((ath for ath, files in self.athlete_mapping.items()
                                             if attempt in files), None)
                    auto = self.attempt_assignments.get(attempt)
                    srt = sims_cache.get(attempt, [])
                    self._build_attr_row(frame, attempt, srt)

                    # tooltip: кандидаты атрибуции с процентами
                    tip = self._attribution_tip(attempt, assigned_athlete, srt)
                    if tip:
                        self._bind_tooltip(label, tip)
                    label.bind('<ButtonRelease-1>', lambda e, a=attempt: self.on_attempt_drag_end(e, a))

                    # Добавляем контекстное меню
                    label.bind('<Button-3>', lambda e, a=attempt: self.show_attempt_context_menu(e, a))

                except Exception as e:
                    self.log(f"Ошибка при создании превью для {attempt}: {str(e)}")

                # Обновляем позицию для следующего превью
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

        # --- Кандидаты ручного режима (ещё не нарезанные) ---
        pending = [c for c in self.candidates if c['status'] == 'pending']
        self.candidate_vars.clear()
        if pending:
            sep_frame = tk.Frame(self.attempts_scrollable_frame)
            sep_frame.grid(row=row + 1, column=0, columnspan=max_cols, pady=(12, 2))
            tk.Label(sep_frame,
                     text=f"—— Кандидаты ({len(pending)}): снимите галочки с ненужных, "
                          f"затем «Нарезать выбранное» ——",
                     fg="#555555").pack(side=tk.LEFT)
            # Очистка ненарезанных кандидатов (для пересканирования с нуля)
            clear_btn = tk.Button(sep_frame, text="🧹", width=3, bd=1,
                                  command=self.clear_pending_candidates)
            clear_btn.pack(side=tk.LEFT, padx=8)
            self._attach_tooltip(clear_btn, "Удалить всех ненарезанных кандидатов\n"
                                            "(пересканировать с нуля)")
            crow, ccol = row + 2, 0
            for cand in pending:
                frame = tk.Frame(self.attempts_scrollable_frame)
                frame.grid(row=crow, column=ccol, padx=5, pady=5)

                img_label = None
                thumb = cand.get('thumbnail')
                if thumb and os.path.exists(thumb):
                    try:
                        image = Image.open(thumb)
                        image = image.resize((200, 150), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        img_label = tk.Label(frame, image=photo)
                        img_label.image = photo
                        img_label.pack()
                    except Exception as e:
                        self.log(f"Ошибка превью кандидата: {e}")

                name_frame = tk.Frame(frame)
                name_frame.pack()

                var = tk.BooleanVar(value=bool(cand.get('selected')))
                checkbox = tk.Checkbutton(name_frame, variable=var,
                                          command=lambda c=cand, v=var: self.on_candidate_checkbox_change(c, v))
                checkbox.pack(side=tk.LEFT, padx=(0, 5))

                # Компактная подпись: имя без расширения, диапазон, оценка объёма выхлопа
                stem = os.path.splitext(os.path.basename(cand['file']))[0]
                est = self._estimate_output_bytes(
                    cand['file'],
                    cand['end'] - cand['start'] + self.processing_params['attempt_start_padding']
                    + self.processing_params['attempt_end_padding']
                    + float(self.processing_params.get('scan_interval', 2.5)) + 1.0)
                size_txt = f" ≈{self._fmt_size(est)}" if est > 0 else ""
                src_missing = not cand.get('file') or \
                    not os.path.exists(self.get_absolute_path(cand['file']))
                if src_missing:
                    size_txt += "  (нет исходника)"
                name_label = tk.Label(name_frame,
                                      text=f"{stem} {cand['start']:.0f}–{cand['end']:.0f}s{size_txt}",
                                      fg='#d32f2f' if src_missing else None)
                name_label.pack(side=tk.LEFT)

                # Двойной клик - видео кандидата из оригинала; правый клик - фотопревью
                for w in (img_label, name_label):
                    if w is not None:
                        w.bind('<Double-Button-1>',
                               lambda e, c=cand: self.show_candidate_video(c))
                        w.bind('<Button-3>',
                               lambda e, c=cand: self.show_candidate_preview(c))

                self.candidate_vars[self._candidate_key(cand)] = var

                ccol += 1
                if ccol >= max_cols:
                    ccol = 0
                    crow += 1

    def on_attempt_checkbox_change(self, attempt, var):
        """Обработчик изменения состояния чекбокса попытки"""
        if var.get():
            self.selected_attempts.add(attempt)
        else:
            self.selected_attempts.discard(attempt)
        
        # Обновляем счетчик
        self._update_selected_count()

    def _update_selected_count(self):
        """Счётчик выбранных попыток + суммарный размер файлов на диске."""
        total = 0
        for name in self.selected_attempts:
            try:
                total += os.path.getsize(os.path.join(self.output_folder, name))
            except OSError:
                pass
        if total >= 1024 ** 3:
            size = f"{total / 1024 ** 3:.1f} ГБ"
        elif total >= 1024 ** 2:
            size = f"{total / 1024 ** 2:.0f} МБ"
        else:
            size = f"{total / 1024:.0f} КБ"
        self.selected_count_label.config(
            text=f"Выбрано: {len(self.selected_attempts)} ({size})")
        btn = getattr(self, 'export_button', None)
        if btn is not None:
            btn.config(state=tk.NORMAL if self.selected_attempts else tk.DISABLED)

    def export_selected_attempts(self):
        """Кнопка 'Экспорт в...': выбор папки и копирование выбранных
        клипов (+ превью jpg) в воркере с прогрессом в логе."""
        if not self.selected_attempts:
            return
        dst = filedialog.askdirectory(title="Куда экспортировать попытки")
        if not dst:
            return
        names = sorted(self.selected_attempts)
        total = sum((os.path.getsize(os.path.join(self.output_folder, n))
                     for n in names
                     if os.path.exists(os.path.join(self.output_folder, n))), 0)
        size = f"{total / 1024 ** 3:.1f} ГБ" if total >= 1024 ** 3 \
            else f"{total / 1024 ** 2:.0f} МБ"
        if not messagebox.askyesno(
                "Экспорт",
                f"Скопировать {len(names)} видео (+превью, {size}) в:\n{dst}?"):
            return
        self._export_names = names
        self._export_dst = dst
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._start_worker(self._export_worker)

    def _export_worker(self):
        """Копирование выбранных попыток: mp4 + jpg в папку назначения."""
        names = getattr(self, '_export_names', [])
        dst = getattr(self, '_export_dst', None)
        try:
            if not names or not dst:
                return
            done = 0
            for i, name in enumerate(names):
                if not self.processing:
                    break
                src_mp4 = os.path.join(self.output_folder, name)
                try:
                    shutil.copy2(src_mp4, os.path.join(dst, name))
                    src_jpg = os.path.join(self.output_folder,
                                           os.path.splitext(name)[0] + '.jpg')
                    if os.path.exists(src_jpg):
                        shutil.copy2(src_jpg, os.path.join(dst, os.path.basename(src_jpg)))
                except Exception as e:
                    self.log(f"  Ошибка копирования {name}: {e}", logging.ERROR)
                    continue
                done += 1
                self.log(f"Экспорт {done}/{len(names)}: {name}")
            self.log(f"Экспорт завершён: {done} из {len(names)} → {dst}")
            if done:
                self._reveal_in_finder(dst)
        except Exception as e:
            self.log(f"Ошибка экспорта: {e}", logging.ERROR)
        finally:
            self.processing = False
            self.root.after(0, lambda: self._after_worker_ui(False))

    def _reveal_in_finder(self, path):
        """Открывает папку в системном файловом менеджере."""
        try:
            if sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            elif os.name == 'nt':
                os.startfile(path)
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception as e:
            self.log(f"Не удалось открыть папку {path}: {e}", logging.WARNING)

    def _delete_attempt_files(self, names):
        """Фактическое удаление попыток: клипы, превью, маппинг, авто-привязки,
        рейтинги, профили. Без диалогов и пересборки UI."""
        changed_assign = False
        for attempt in names:
            video_path = os.path.join(self.output_folder, attempt)
            thumbnail_path = os.path.join(
                self.output_folder, f"{os.path.splitext(attempt)[0]}.jpg")
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
            except Exception as e:
                self.log(f"Ошибка при удалении {attempt}: {str(e)}", logging.ERROR)
                continue
            for athlete_name, attempts_list in self.athlete_mapping.items():
                if attempt in attempts_list:
                    attempts_list.remove(attempt)
            if self.attempt_assignments.pop(attempt, None) is not None:
                changed_assign = True
            self.attempt_ratings.pop(attempt, None)
            self.selected_attempts.discard(attempt)
            profile_store = self._load_profile_store()
            if attempt in profile_store:
                del profile_store[attempt]
                self._save_profile_store(profile_store)
            self.log(f"Удален файл: {attempt}")
        if changed_assign:
            self._save_assignments()

    def delete_attempt_file(self, attempt):
        """Удаление одной попытки (двухфазная кнопка в плеере) + обновление UI"""
        self._delete_attempt_files([attempt])
        self.save_athlete_mapping()
        self.save_ratings()
        self.update_attempt_thumbnails()
        self.update_athlete_list()

    def delete_selected_attempts(self):
        """Удаляет выбранные попытки"""
        if not self.selected_attempts:
            messagebox.showwarning("Предупреждение", "Не выбрано ни одной попытки для удаления")
            return

        # Запрашиваем подтверждение
        count = len(self.selected_attempts)
        result = messagebox.askyesno("Подтверждение",
                                   f"Вы уверены, что хотите удалить {count} попытку(и)?\n"
                                   "Это действие нельзя отменить.")

        if result:
            names = list(self.selected_attempts)
            self._delete_attempt_files(names)
            self.selected_attempts.clear()

            # Сохраняем почищенные маппинг и рейтинги
            self.save_athlete_mapping()
            self.save_ratings()

            # Обновляем счетчик
            self._update_selected_count()

            # Обновляем интерфейс
            self.update_attempt_thumbnails()
            self.update_athlete_list()

            messagebox.showinfo("Успех", f"Удалено {len(names)} попыток")

    def select_all_attempts(self):
        """Выбирает все отображаемые попытки"""
        # Получаем отфильтрованные попытки
        filtered_files = self.get_filtered_attempts()
        attempts = [os.path.basename(f) for f in filtered_files]
        
        # Выбираем все попытки
        self.selected_attempts = set(attempts)
        
        # Обновляем чекбоксы
        for attempt, checkbox_var in self.attempt_checkboxes.items():
            if attempt in self.selected_attempts:
                checkbox_var.set(True)
        
        # Обновляем счетчик
        self._update_selected_count()

    def deselect_all_attempts(self):
        """Снимает выделение со всех попыток"""
        # Очищаем выбранные попытки
        self.selected_attempts.clear()
        
        # Обновляем чекбоксы
        for checkbox_var in self.attempt_checkboxes.values():
            checkbox_var.set(False)
        
        # Обновляем счетчик
        self._update_selected_count()

    def toggle_attempt_rating(self, attempt, rating_type):
        """Переключает рейтинг попытки"""
        if attempt not in self.attempt_ratings:
            self.attempt_ratings[attempt] = {'up': False, 'down': False}
        
        # Переключаем состояние
        self.attempt_ratings[attempt][rating_type] = not self.attempt_ratings[attempt][rating_type]
        
        # Сохраняем рейтинги
        self.save_ratings()
        
        # Обновляем интерфейс
        self.update_attempt_thumbnails()

    def update_rating_button_style(self, up_button, down_button, rating_state):
        """Обновляет стиль кнопок рейтинга"""
        if rating_state['up']:
            up_button.config(relief=tk.SUNKEN, bg=self.button_face_color, fg='black',
                           text='👍', font=('Arial', 9, 'normal'),
                           bd=2, highlightbackground='blue')
        else:
            up_button.config(relief=tk.RAISED, bg=self.button_face_color, fg='black',
                           text='👍', font=('Arial', 9, 'normal'),
                           bd=1, highlightbackground=self.button_face_color)

        if rating_state['down']:
            down_button.config(relief=tk.SUNKEN, bg=self.button_face_color, fg='black',
                             text='👎', font=('Arial', 9, 'normal'),
                             bd=2, highlightbackground='blue')
        else:
            down_button.config(relief=tk.RAISED, bg=self.button_face_color, fg='black',
                             text='👎', font=('Arial', 9, 'normal'),
                             bd=1, highlightbackground=self.button_face_color)

    def toggle_rating_filter(self, rating_type):
        """Переключает фильтр по рейтингу"""
        if rating_type in self.active_rating_filters:
            self.active_rating_filters.remove(rating_type)
        else:
            self.active_rating_filters.add(rating_type)
        
        # Обновляем стиль кнопок фильтров
        self.update_filter_button_styles()
        
        # Обновляем отображение попыток
        self.update_attempt_thumbnails()

    def update_filter_button_styles(self):
        """Обновляет стиль кнопок фильтров"""
        if "up" in self.active_rating_filters:
            self.filter_up_button.config(relief=tk.SUNKEN, bg=self.button_face_color, fg='black',
                                       text='👍', font=('Arial', 11, 'normal'),
                                       bd=2, highlightbackground='blue')
        else:
            self.filter_up_button.config(relief=tk.RAISED, bg=self.button_face_color, fg='black',
                                       text='👍', font=('Arial', 11, 'normal'),
                                       bd=2, highlightbackground=self.button_face_color)

        if "down" in self.active_rating_filters:
            self.filter_down_button.config(relief=tk.SUNKEN, bg=self.button_face_color, fg='black',
                                         text='👎', font=('Arial', 11, 'normal'),
                                         bd=2, highlightbackground='blue')
        else:
            self.filter_down_button.config(relief=tk.RAISED, bg=self.button_face_color, fg='black',
                                         text='👎', font=('Arial', 11, 'normal'),
                                         bd=2, highlightbackground=self.button_face_color)

    def on_attempt_double_click(self, event, attempt):
        """Обработчик двойного клика по попытке"""
        # Отменяем таймер перетаскивания
        if hasattr(self, 'drag_timer'):
            self.root.after_cancel(self.drag_timer)
            delattr(self, 'drag_timer')

        # Встроенный видеоплеер с начала попытки
        self.show_attempt_video(attempt)

    def _preview_slowmo_factor(self):
        """Коэффициент слоумо для превью: источник истины - чекбокс+комбо"""
        return self._ui_slowmo_factor()

    def next_attempt_after(self, key):
        """Следующая попытка (клип) по порядку плиток, или None"""
        if not key:
            return None
        names = sorted(self.attempt_checkboxes.keys())
        if key not in names:
            return None
        i = names.index(key)
        return names[i + 1] if i + 1 < len(names) else None

    def prev_attempt_before(self, key):
        """Предыдущая попытка (клип) по порядку плиток, или None"""
        if not key:
            return None
        names = sorted(self.attempt_checkboxes.keys())
        if key not in names:
            return None
        i = names.index(key)
        return names[i - 1] if i > 0 else None

    def show_attempt_video(self, attempt):
        """Двойной клик по попытке: нарезанный клип (строка-имя) или оригинал
        с начала попытки (AttemptInfo) - встроенный плеер."""
        if isinstance(attempt, str):
            src = self.get_absolute_path(attempt)
            if not os.path.exists(src):
                messagebox.showinfo("Превью", f"Клип не найден: {attempt}")
                return
            # нарезанные клипы хранятся в обычной скорости
            VideoPreviewPopup(self, src, 0.0, f"Попытка {attempt}", factor=1,
                              attempt=attempt)
            return
        src = self.get_absolute_path(attempt.source_video)
        if not os.path.exists(src):
            messagebox.showinfo("Превью", "Исходный файл не найден")
            return
        factor = self._preview_slowmo_factor()
        start_real = max(0.0, float(attempt.start) - 1.0)
        VideoPreviewPopup(self, src,
                          start_real * factor if factor > 1 else start_real,
                          f"Попытка {attempt.number:04d}", factor=factor,
                          attempt=attempt)

    def show_candidate_video(self, cand):
        """Двойной клик по кандидату: видео из оригинала с позиции кандидата
        (встроенный плеер, hotkeys, Esc). Фотопревью остаётся на правый клик."""
        src = self.get_absolute_path(cand.get('file', ''))
        if not cand.get('file') or not os.path.exists(src):
            messagebox.showinfo("Превью", "Исходный файл не найден")
            return
        factor = self._preview_slowmo_factor()
        start_real = max(0.0, float(cand.get('start', 0)) - 1.0)
        stem = os.path.splitext(os.path.basename(cand['file']))[0]
        VideoPreviewPopup(self, src,
                          start_real * factor if factor > 1 else start_real,
                          f"{stem} {cand['start']:.0f}–{cand['end']:.0f}s",
                          factor=factor)

    def show_candidate_preview(self, cand):
        """Превью кандидата в сохранённом (полном) размере - попапом.
        Закрывается кликом или Esc."""
        thumb = cand.get('thumbnail')
        if not thumb or not os.path.exists(thumb):
            messagebox.showinfo("Превью", "Превью кандидата не найдено")
            return
        try:
            img = Image.open(thumb)
        except Exception as e:
            self.log(f"Не удалось открыть превью: {e}")
            return
        # Не выходим за пределы экрана
        max_w = self.root.winfo_screenwidth() - 80
        max_h = self.root.winfo_screenheight() - 120
        if img.width > max_w or img.height > max_h:
            img = img.copy()
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        top = tk.Toplevel(self.root)
        stem = os.path.splitext(os.path.basename(cand['file']))[0]
        top.title(f"{stem} {cand['start']:.0f}–{cand['end']:.0f}s")
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo, cursor="hand2")
        label.image = photo
        label.pack()
        top.resizable(False, False)
        for seq in ('<Button-1>', '<Escape>', '<Return>'):
            top.bind(seq, lambda e: top.destroy())
        label.focus_set()
        top.grab_set()

    def _attach_tooltip(self, widget, text):
        """Простой всплывающий хинт при наведении"""
        tip = {'tw': None}

        def enter(_):
            if tip['tw']:
                return
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{widget.winfo_rootx() + 12}+"
                           f"{widget.winfo_rooty() + widget.winfo_height() + 4}")
            tk.Label(tw, text=text, bg="#ffffe0", relief="solid", borderwidth=1,
                     justify=tk.LEFT, font=('Arial', 9)).pack()
            tip['tw'] = tw

        def leave(_):
            if tip['tw']:
                tip['tw'].destroy()
                tip['tw'] = None

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def show_attempt_file(self, attempt):
        """Открывает папку с файлом попытки и выделяет его"""
        if os.path.exists(attempt):
            # Для macOS
            subprocess.run(['open', '-R', attempt])
            # Для Windows
            # subprocess.run(['explorer', '/select,', attempt])
            # Для Linux
            # subprocess.run(['xdg-open', os.path.dirname(attempt)])

    def log(self, message, level=logging.INFO):
        """Добавляет сообщение в лог"""
        if self.logger:
            self.logger.log(level, message)

    def on_click(self, event):
        if self.roi_mode == "rectangle":
            self.start_x = event.x
            self.start_y = event.y
            self.dragging = True
        elif self.roi_mode == "polygon":
            if not self.drawing_polygon:
                # Начинаем рисовать новый многоугольник
                self.roi_points = []
                self.drawing_polygon = True
                self.canvas.delete("roi_polygon")
                self.canvas.delete("roi_points")
            
            # Добавляем точку
            self.roi_points.append((event.x, event.y))
            
            # Рисуем точку
            self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                                  fill="red", tags="roi_points")
            
            # Рисуем линии между точками
            if len(self.roi_points) > 1:
                prev_point = self.roi_points[-2]
                self.canvas.create_line(prev_point[0], prev_point[1], event.x, event.y, 
                                      fill="red", width=2, tags="roi_polygon")

    def on_drag(self, event):
        if self.dragging and self.roi_mode == "rectangle":
            self.end_x = event.x
            self.end_y = event.y
            self.canvas.delete("roi_rectangle")
            self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="roi_rectangle")

    def on_release(self, event):
        if self.dragging and self.roi_mode == "rectangle":
            self.dragging = False
            self.end_x = event.x
            self.end_y = event.y
            self.roi = (100 * self.start_x/ self.canvas_width, 100 * self.start_y /CANVAS_HEIGHT, 100 * self.end_x / self.canvas_width, 100* self.end_y/ CANVAS_HEIGHT)
            self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="roi_rectangle")

    def finish_polygon(self, event):
        """Завершает рисование многоугольника"""
        if self.roi_mode == "polygon" and self.drawing_polygon and len(self.roi_points) >= 3:
            # Замыкаем многоугольник
            if len(self.roi_points) > 2:
                first_point = self.roi_points[0]
                last_point = self.roi_points[-1]
                self.canvas.create_line(last_point[0], last_point[1], first_point[0], first_point[1], 
                                      fill="red", width=2, tags="roi_polygon")
            
            # Сохраняем ROI в процентах
            self.roi = [[100 * x / self.canvas_width, 100 * y / CANVAS_HEIGHT] for x, y in self.roi_points]
            self.drawing_polygon = False
            self.log(f"Многоугольная ROI создана с {len(self.roi_points)} точками")

    def clear_roi(self):
        """Очищает текущую ROI"""
        self.canvas.delete("roi_rectangle")
        self.canvas.delete("roi_polygon")
        self.canvas.delete("roi_points")
        self.roi = None
        self.roi_points = []
        self.drawing_polygon = False
        self.dragging = False

    def switch_roi_mode(self, mode):
        """Переключает режим ROI между прямоугольником и многоугольником"""
        self.roi_mode = mode
        self.clear_roi()
        if mode == "rectangle":
            self.label_roi.config(text="Выберите прямоугольную область интереса:")
        else:
            self.label_roi.config(text="Кликните по точкам многоугольника. Двойной клик завершает:")

    def on_file_select(self, event):
        """Обработчик выбора файла в списке"""
        selection = self.listbox_files.curselection()
        if selection:
            index = selection[0]
            if 0 <= index < len(self.selected_files):
                self.current_video_for_roi = self.selected_files[index]
                self.current_frame_position = 0.5  # Сбрасываем позицию на середину
                self.create_canvas_from_video(self.current_video_for_roi)

    def show_next_frame(self):
        """Показывает следующий кадр из текущего видео"""
        if not self.current_video_for_roi:
            if self.selected_files:
                self.current_video_for_roi = self.selected_files[0]
            else:
                messagebox.showwarning("Предупреждение", "Не выбраны файлы для обработки.")
                return

        # Перебираем позиции с шагом 10%
        self.current_frame_position = (self.current_frame_position + 0.1) % 1.0
        if self.current_frame_position < 0.1:  # Если прошли полный круг, начинаем с 10%
            self.current_frame_position = 0.1

        self.create_canvas_from_video(self.current_video_for_roi)

    def _ui_slowmo_factor(self):
        """Коэффициент слоумо из UI: чекбокс + комбобокс ('1/2','1/4','1/8')"""
        if self.slowmo_var and self.slowmo_var.get():
            try:
                return max(2, int(self.slowmo_combo_var.get().split('/')[1]))
            except (IndexError, ValueError):
                return 4
        return 1

    def _set_slowmo_ui(self, factor):
        """Выставляет чекбокс+комбобокс слоумо по коэффициенту (автодетект)"""
        factor = max(1, int(factor))
        self.slowmo_var.set(factor > 1)
        if factor > 1:
            self.slowmo_combo_var.set(f"1/{factor}")
        self.slowmo_combo.config(state='readonly' if factor > 1 else 'disabled')

    def on_slowmo_toggle(self):
        """Обработчик переключения чекбокса слоумо"""
        factor = self._ui_slowmo_factor()
        self.slowmo_combo.config(state='readonly' if factor > 1 else 'disabled')
        if factor > 1:
            self.log(f"Режим слоумо включён: видео 1/{factor} скорости "
                     "(например, 96 к/с как 24 к/с). Попытки и параметры "
                     "считаются в реальных секундах, вырезанные файлы будут "
                     "обычной скорости.")
        else:
            self.log("Режим слоумо выключен.")

    def select_files(self):
        initial_dir = find_default_video_folder()
        files = filedialog.askopenfilenames(
            title="Выберите видеофайлы",
            filetypes=[("Video Files", "*.MTS *.MP4 *.AVI")],
            initialdir=initial_dir)
        if not files:
            return
        self.selected_files = list(files)
        self._render_file_list()
        self.save_processing_config()
        self.update_folder_status()

        # Устанавливаем первое видео как текущее для ROI
        if self.selected_files:
            self.current_video_for_roi = self.selected_files[0]
            self.current_frame_position = 0.5
            self.create_canvas_from_video(self.current_video_for_roi)

            # Пытаемся автоматически определить слоумо по метаданным
            # (для Panasonic GH4 VFR метаданных нет - тогда вручную чекбоксом)
            factor = detect_slowmo_factor(self.selected_files[0])
            self._set_slowmo_ui(factor)
            if factor > 1:
                self.log(f"Обнаружено слоумо-видео (замедление 1/{factor}) - включён режим слоумо")
            else:
                self.log("Слоумо по метаданным не обнаружено. Если это слоумо 96 к/с (GH4) - "
                         "включите галочку \"Слоумо\" вручную")

    def change_output_folder(self):
        """Выбор произвольной рабочей папки через стандартный диалог ОС"""
        initial = self.output_folder if os.path.isdir(self.output_folder) \
            else os.path.expanduser("~/Desktop/FreestyleParser")
        folder = filedialog.askdirectory(initialdir=initial)
        if folder:
            self._set_output_folder(folder, f"Выбрана произвольная папка: {folder}")

    def _sync_params_from_ui(self):
        """Считывает значения параметров из UI в processing_params"""
        for key, var in self.param_vars.items():
            try:
                val = float(var.get())
                self.processing_params[key] = val
            except ValueError:
                self.log(f"Некорректное значение параметра {key}: {var.get()}")
                var.set(str(self.processing_params[key]))

    def _sync_ui_from_params(self):
        """Обновляет UI из processing_params"""
        for key, var in self.param_vars.items():
            var.set(str(self.processing_params[key]))

    def generate_thumbnail(self, video_path, output_path, roi=None):
        """Создает превью для видео с учетом ROI"""
        try:
            if roi:
                # Сначала получаем полный кадр из видео
                temp_path = output_path + ".temp.jpg"
                cmd = [
                    get_ffmpeg_path(), '-ss', '3.0',  # Берем кадр на 3 секунде
                    '-i', self.get_absolute_path(video_path),
                    '-vframes', '1',
                    '-q:v', '2',  # Максимальное качество JPEG
                    '-y', temp_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Получаем размеры оригинального кадра
                img = Image.open(temp_path)
                orig_width, orig_height = img.size

                # Проверяем формат ROI
                if len(roi) == 4:  # Прямоугольник (две точки: x1,y1,x2,y2)
                    x1, y1, x2, y2 = roi
                    crop_x = int(x1 * orig_width / 100)
                    crop_y = int(y1 * orig_height / 100)
                    crop_w = int((x2 - x1) * orig_width / 100)
                    crop_h = int((y2 - y1) * orig_height / 100)
                    if crop_h == 0 or crop_w == 0:
                        self.log(f"Ошибка вырезания скриншота roi={roi}; wxh ={orig_width}x{orig_height}")
                    # Вырезаем ROI из оригинального кадра
                    cmd = [
                        get_ffmpeg_path(),
                        '-i', temp_path,
                        '-vf', f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={THUMB_X}:-1',  # Сначала crop, потом scale
                        '-q:v', '2',
                        '-y', output_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                elif len(roi) > 4:  # Многоугольник (новый формат)
                    # Для многоугольника используем Python для обработки
                    import cv2
                    import numpy as np
                    
                    # Загружаем изображение
                    frame = cv2.imread(temp_path)
                    
                    # Для превью показываем полный кадр без маски
                    # Маска применяется только при обработке для детекции
                    # Здесь мы просто используем полный кадр для красивого превью

                    # Полный кадр - единообразно с превью при обработке
                    cv2.imwrite(output_path, frame)

                # Удаляем временный файл
                if os.path.exists(temp_path):
                    try:
                        # Даем небольшую задержку для освобождения файла
                        time.sleep(0.1)
                        os.remove(temp_path)
                    except PermissionError:
                        self.log(f"Не удалось удалить временный файл {temp_path} - файл занят")
                    except Exception as e:
                        self.log(f"Ошибка при удалении временного файла: {e}")
            else:
                cmd = [
                    get_ffmpeg_path(), '-ss', '2.5',
                    '-i', self.get_absolute_path(video_path),
                    '-vframes', '1',
                    '-vf', f'scale={THUMB_X}:-1',
                    '-q:v', '2',
                    '-y', output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Ошибка создания скриншота: {e}")
            return False

    def smart_thumbnail(self, video_path, output_path):
        """Превью попытки: кадр, где каякер ближе к центру кадра и крупнее.
        Прогоняет YOLO по сэмплам клипа (~2/с, до 120 сэмплов)."""
        try:
            if self._thumb_model is None:
                self._thumb_model = load_yolo_model()
            model = self._thumb_model
            cap = cv2.VideoCapture(self.get_absolute_path(video_path))
            if not cap.isOpened():
                return False
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = n_frames / fps if fps > 0 else 0.0
            step_s = 0.5
            n_samples = min(int(duration / step_s) + 1, 120)
            step = max(1, round(n_frames / max(n_samples, 1)))
            best = None  # (score, frame)
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi % step == 0:
                    small = cv2.resize(frame, None, fx=0.4, fy=0.4)
                    res = model(small, verbose=False)[0]
                    H, W = frame.shape[:2]
                    for box in res.boxes:
                        if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.4:
                            continue
                        x1, y1, x2, y2 = [c / 0.4 for c in box.xyxy[0].cpu().numpy()]
                        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                        dist = ((abs(cx - 0.5) * 2) ** 2 + (abs(cy - 0.5) * 2) ** 2) ** 0.5
                        area = max((x2 - x1) / W * (y2 - y1) / H, 1e-6)
                        score = float(box.conf[0]) * (1.0 - dist) * (area ** 0.25)
                        if best is None or score > best[0]:
                            best = (score, frame.copy())
                fi += 1
            cap.release()
            if best is None:
                return self.generate_thumbnail(os.path.basename(video_path),
                                               output_path, self.roi)
            # полный кадр FHD - как при обработке (UI сам масштабирует для показа)
            cv2.imwrite(output_path, best[1])
            return True
        except Exception as e:
            self.log(f"Ошибка умного превью {video_path}: {e}", logging.ERROR)
            return False

    def regenerate_all_thumbnails(self):
        """Пересоздаёт превью всех попыток через smart_thumbnail (в воркере)."""
        try:
            if not os.path.exists(self.output_folder):
                return
            files = sorted(f for f in os.listdir(self.output_folder)
                           if f.endswith(".mp4"))
            for i, name in enumerate(files):
                if not self.processing:
                    break
                jpg = os.path.join(self.output_folder,
                                   f"{os.path.splitext(name)[0]}.jpg")
                if os.path.exists(jpg):
                    os.remove(jpg)
                self.smart_thumbnail(name, jpg)
                self.log(f"Превью {i + 1}/{len(files)}: {name}")
        except Exception as e:
            self.log(f"Ошибка регенерации превью: {e}", logging.ERROR)
        finally:
            self.processing = False
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_attempt_thumbnails)

    def start_thumbnail_regeneration(self):
        """Кнопка 'Обновить превью': блокирует кнопки и запускает воркер."""
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._start_worker(self.regenerate_all_thumbnails)

    def start_bank_detection(self):
        """Кнопка 'Детекция по банкам': автоатрибуция непривязанных попыток
        без пересканирования видео (для текущей и прошлых папок)."""
        if self.processing:
            messagebox.showinfo("Информация", "Идёт обработка - дождитесь завершения.")
            return
        if not any(files for files in self.athlete_mapping.values()):
            messagebox.showinfo(
                "Информация",
                "Сначала привяжите несколько попыток к атлетам (кнопка ✓ или селектор):\n"
                "банки строятся из подтверждённых привязок.")
            return
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._start_worker(self._bank_detection_worker)

    def _profile_from_clip(self, name):
        """Короткая схема: профиль попытки из готового клипа (YOLO + OSNet
        на 2-3 лучших кропах)."""
        import attempts_v2
        if self._thumb_model is None:
            self._thumb_model = load_yolo_model()
        if self._embedder is None:
            self._embedder = attempts_v2.Embedder(attempts_v2.default_device())
        return attempts_v2.profile_from_clip(self.get_absolute_path(name),
                                             model=self._thumb_model,
                                             embedder=self._embedder)

    def _bank_detection_worker(self):
        """Детекция атлетов по банкам: профили попыток берутся из
        attempt_profiles.npz, отсутствующие досчитываются из клипов
        (2-3 кадра), попытки предлагаются по мере обработки - плитки
        обновляются постепенно (need_update_attempts + periodic_update)."""
        try:
            import attempts_v2
            self._prune_athlete_mapping()
            store = self._load_profile_store()
            clips = sorted(f for f in os.listdir(self.output_folder)
                           if f.endswith(".mp4") and f[:4].isdigit())
            missing = [n for n in clips
                       if n not in store
                       or float(np.linalg.norm(
                           store[n]['vest'][attempts_v2.VEST_DIM:])) < 1e-6]
            if missing:
                self.log(f"Попыток без профиля: {len(missing)} - короткая схема "
                         f"(2-3 кадра из клипа на попытку)")
            # прошлые провизорные привязки пересматриваем вместе с новыми
            with self._assignments_lock:
                if self.attempt_assignments:
                    self.attempt_assignments = {}
                    self._save_assignments()
                    self.log("Прошлые авто-привязки сняты - пересматриваю по банкам")
            self.need_update_attempts = True
            # профили подтверждённых (маппинг) заполняем ПЕРВЫМИ - из них
            # строятся банки; иначе банки пусты и круг не размыкается
            mapped_names = {os.path.basename(f)
                            for files in self.athlete_mapping.values()
                            for f in files}
            for name in clips:
                if name not in mapped_names:
                    continue
                if name in store and float(np.linalg.norm(
                        store[name]['vest'][attempts_v2.VEST_DIM:])) >= 1e-6:
                    continue
                prof = self._profile_from_clip(name)
                if prof is None:
                    self.log(f"  {name}: атлет в клипе не найден (нужен для банка)")
                    continue
                store[name] = prof
                self._save_profile_store(store)
                self.log(f"  Профиль из клипа (банк): {name}")
            banks = self._athlete_banks(store)
            if not banks:
                self.log("Банки пусты: нет подтверждённых атлетов и базовых банков. "
                         "Привяжите попытки вручную или подтвердите сегмент - "
                         "банк построится из их профилей", logging.WARNING)
                return
            self.log(f"Банки атлетов: {', '.join(banks.keys())}")
            assigned_names = {os.path.basename(f)
                              for files in self.athlete_mapping.values()
                              for f in files}
            n_auto = 0
            for name in clips:
                if not self.processing:
                    break
                with self._assignments_lock:
                    if name in assigned_names or name in self.attempt_assignments:
                        continue
                if name not in store or \
                        float(np.linalg.norm(
                            store[name]['vest'][attempts_v2.VEST_DIM:])) < 1e-6:
                    prof = self._profile_from_clip(name)
                    if prof is None:
                        self.log(f"  {name}: атлет в клипе не найден - пропущена")
                        continue
                    store[name] = prof
                    self._save_profile_store(store)
                    self.log(f"  Профиль из клипа: {name}")
                with self._assignments_lock:
                    if self._try_auto_assign(store, name, banks):
                        n_auto += 1
            self.log(f"Детекция по банкам: предложено {n_auto}")
        except Exception as e:
            self.log(f"Ошибка детекции по банкам: {e} {traceback.format_exc()}",
                     logging.ERROR)
        finally:
            self.processing = False
            self.need_update_attempts = True
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_athlete_list)

    def _on_selector_assign(self, attempt, choice):
        """Выбор атлета в селекторе на плитке превью."""
        if choice == '➕ Новый / поиск…':
            name = self._ask_athlete(self.root)
            if not name:
                return
            self.assign_attempt_to_athlete(attempt, name)
            return
        if choice == '— снять':
            self.remove_assignment(attempt)
        else:
            self.assign_attempt_to_athlete(attempt, choice)

    def _attribution_tip(self, attempt, assigned_athlete, srt):
        """Текст tooltip: кандидаты атрибуции с процентами (или текущий статус)."""
        if assigned_athlete:
            return f"{assigned_athlete} — подтверждено"
        if not srt:
            return None
        tip = ', '.join(f'{a} {s * 100:.0f}%' for a, s in srt[:2])
        auto = self.attempt_assignments.get(attempt)
        if auto and len(srt) > 1:
            tip += f"\nвторой кандидат: {srt[1][0]} {srt[1][1] * 100:.0f}%"
        return tip

    def _bind_tooltip(self, widget, text):
        """Простой tooltip при наведении."""
        tip_win = {'w': None}

        def enter(_):
            if tip_win['w']:
                return
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            w = tk.Toplevel(widget)
            w.wm_overrideredirect(True)
            w.wm_geometry(f"+{x}+{y}")
            tk.Label(w, text=text, bg='#ffffe0', relief=tk.SOLID, bd=1,
                     font=('Arial', 9), justify=tk.LEFT).pack()
            tip_win['w'] = w

        def leave(_):
            if tip_win['w']:
                tip_win['w'].destroy()
                tip_win['w'] = None

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    @staticmethod
    def _short(s, n=16):
        """Обрезка длинного имени с многоточием (вёрстка плиток/строк)"""
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + '…'

    def _athlete_chip(self, parent, current, choices, on_pick,
                      bg='#b0b0b0', fg='white', on_open=None):
        """Цветной чип-селектор атлета: tk.Label + tk.Menu. OptionMenu не
        используем - на macOS aqua он игнорирует bg. Один элемент вместо
        пары 'статус + селектор' - вдвое компактнее. on_open - действие при
        открытии меню (в плеере: пауза)."""
        chip = tk.Label(parent, text=self._short(current or '— выбрать —', 14),
                        bg=bg, fg=fg, cursor='hand2', font=('Arial', 8),
                        padx=4, pady=1)
        menu = tk.Menu(chip, tearoff=0, font=('Arial', 8))
        for ch in choices:
            menu.add_command(label=ch, command=lambda c=ch: on_pick(c))
        chip.bind('<Button-1>', lambda e: (
            on_open() if on_open else None,
            menu.tk_popup(chip.winfo_rootx(),
                          chip.winfo_rooty() + chip.winfo_height())))
        return chip

    def _athlete_names(self):
        """Все известные имена: маппинг папки + личный реестр + базовые банки."""
        names = list(self.athlete_mapping)
        names += [n for n in self._registry_banks() if n not in names]
        names += [n for n in self._base_banks() if n not in names]
        return names

    def _ask_athlete(self, parent):
        """Диалог 'Новый/Поиск': живые подсказки из всех банков (по началу
        имени), лупа - поиск по вхождению, внизу явная кнопка 'Создать'.
        Возвращает имя или None."""
        names = self._athlete_names()
        win = tk.Toplevel(parent)
        win.title("Атлет: выбрать или создать")
        win.transient(parent)
        win.resizable(False, False)
        result = {'name': None}

        top = tk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=(8, 2))
        ent = tk.Entry(top, font=('Arial', 11))
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
        mode = {'substr': False}
        mag = tk.Label(top, text='🔍', font=('Arial', 11), cursor='hand2')
        mag.pack(side=tk.LEFT, padx=(4, 0))

        lb = tk.Listbox(win, height=8, font=('Arial', 10), exportselection=False)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        bottom = tk.Frame(win)
        bottom.pack(fill=tk.X, padx=8, pady=(2, 8))
        create_btn = tk.Button(bottom, text='➕ Создать', bd=1,
                               font=('Arial', 9), cursor='hand2')
        create_btn.pack(fill=tk.X)

        def refresh(*_):
            q = ent.get().strip().casefold()
            lb.delete(0, tk.END)
            lb.selection_clear(0, tk.END)
            if not q:
                pool = names
            elif mode['substr']:
                pool = [n for n in names if q in n.casefold()]
            else:
                pool = [n for n in names if n.casefold().startswith(q)]
            for n in pool[:20]:
                lb.insert(tk.END, n)
            name = ent.get().strip()
            create_btn.config(text=f"➕ Создать «{self._short(name, 24)}»"
                              if name else '➕ Создать')

        def toggle_mode(_=None):
            mode['substr'] = not mode['substr']
            mag.config(bg='#4a90d9' if mode['substr'] else '')
            refresh()

        def pick(_=None):
            sel = lb.curselection()
            if sel:
                result['name'] = lb.get(sel[0])
            win.destroy()

        def create(_=None):
            name = ent.get().strip()
            if name:
                result['name'] = name
            win.destroy()

        def on_enter(_=None):
            pick() if lb.curselection() else create()

        def on_nav(e):
            if not lb.size():
                return 'break'
            cur = lb.curselection()[0] if lb.curselection() else -1
            nxt = max(0, min(lb.size() - 1, cur + (1 if e.keysym == 'Down' else -1)))
            lb.selection_clear(0, tk.END)
            lb.selection_set(nxt)
            lb.see(nxt)
            return 'break'

        mag.bind('<Button-1>', toggle_mode)
        create_btn.config(command=create)
        lb.bind('<Double-Button-1>', pick)
        lb.bind('<Return>', pick)
        ent.bind('<Return>', on_enter)
        ent.bind('<Down>', on_nav)
        ent.bind('<Up>', on_nav)
        ent.bind('<Escape>', lambda e: win.destroy())
        win.bind('<Escape>', lambda e: win.destroy())
        ent.bind('<KeyRelease>', refresh)

        win.grab_set()
        ent.focus_set()
        refresh()
        win.wait_window(win)
        return result['name']

    def _athlete_selector(self, parent, key, on_pick, expand=False, names=None,
                          on_open=None, on_confirm=None):
        """Общий контрол атрибуции попытки: цветной чип-селектор (зелёный -
        подтверждено, оранжевый - авто-предложение) + кнопка ✓% для авто.
        Один компонент для плиток сетки и плеера. Возвращает (фрейм, refresh)."""
        frame = tk.Frame(parent)
        confirm_fn = on_confirm or (lambda k, b: self.confirm_assignment(k, b))

        def refresh():
            for w in frame.winfo_children():
                w.destroy()
            assigned = next((a for a, files in self.athlete_mapping.items()
                             if key in files), None)
            auto = self.attempt_assignments.get(key)
            choices = ['— снять', '➕ Новый / поиск…'] + (names or self._athlete_names())
            if assigned:
                chip = self._athlete_chip(frame, assigned, choices, on_pick,
                                          bg='#4caf50', on_open=on_open)
                self._bind_tooltip(chip, assigned)
            elif auto:
                chip = self._athlete_chip(frame, auto.get('athlete', '?'),
                                          choices, on_pick, bg='#ff9800',
                                          on_open=on_open)
                self._bind_tooltip(chip, auto.get('athlete', '?'))
                ok = tk.Label(frame, text=f"✓{auto.get('sim', 0) * 100:.0f}%",
                              bg='#4caf50', fg='white',
                              font=('Arial', 8, 'bold'), cursor='hand2',
                              padx=3, pady=1)
                ok.bind('<Button-1>',
                        lambda e, k=key, b=auto.get('athlete'):
                            confirm_fn(k, b))
                ok.pack(side=tk.LEFT)
            else:
                chip = self._athlete_chip(frame, None, choices, on_pick)
            chip.pack(side=tk.LEFT, fill=tk.X, expand=expand)

        refresh()
        return frame, refresh

    def _build_attr_row(self, parent, attempt, srt):
        """Строка атрибуции на плитке (общий компонент _athlete_selector)."""
        attr_frame = tk.Frame(parent)
        attr_frame.pack(fill=tk.X)
        self._attempt_attr_frames[attempt] = attr_frame
        if not (self._athlete_names()
                or self.attempt_assignments.get(attempt)
                or any(attempt in f for f in self.athlete_mapping.values())):
            return attr_frame
        # порядок в меню: кандидаты по похожести, потом остальные
        ordered = [ath for ath, _ in srt]
        ordered += [ath for ath in self.athlete_mapping if ath not in ordered]
        ordered += [ath for ath in self._base_banks() if ath not in ordered]
        sel, _ = self._athlete_selector(
            attr_frame, attempt,
            lambda ch, a=attempt: self._on_selector_assign(a, ch),
            expand=True, names=ordered)
        sel.pack(fill=tk.X)
        return attr_frame

    def _on_segment_select(self, seg, choice):
        """Выбор атлета в селекторе строки сегмента."""
        if choice == '➕ Новый / поиск…':
            name = self._ask_athlete(self.root)
            if not name:
                return
            self.confirm_segment(seg, name)
            return
        if choice == '— снять':
            self.remove_segment_assignment(seg)
        else:
            self.confirm_segment(seg, choice)

    def _select_athlete_tab(self, key):
        """Ручные вкладки панели 'Атлеты / Скан': подсветка + показ фрейма"""
        if not hasattr(self, '_athlete_tab_buttons'):
            return
        for k, btn in self._athlete_tab_buttons.items():
            active = (k == key)
            btn.configure(bg='#ffffff' if active else '#d8d8d8',
                          font=('Arial', 10, 'bold') if active
                          else ('Arial', 10))
        self._athlete_tab_frames[key].tkraise()

    def _raise_scan_tab(self):
        """Автопереключение на вкладку 'Скан' (после завершения скана атлетов)"""
        if hasattr(self, '_athlete_tab_buttons'):
            self._select_athlete_tab('scan')

    def _update_scan_tab_title(self):
        """Заголовок вкладки Скан: счётчик неподтверждённых + статус банка"""
        if not hasattr(self, '_athlete_tab_buttons'):
            return
        pending = sum(1 for s in self.athlete_segments if not s.get('confirmed'))
        btn = self._athlete_tab_buttons.get('scan')
        if btn is not None:
            btn.configure(text=f"🔍 Скан ({pending})" if pending else "🔍 Скан")
        n_base = len(self._base_banks())
        n_personal = len(self._registry_banks())
        self.bank_status_var.set(
            f"Банк: база {n_base} + ваши {n_personal}"
            if (n_base or n_personal) else
            "Банк пуст: подтвердите атлета -\nдальше программа узнает сама")

    def update_segments_panel(self):
        """Вкладка 'Скан': строки сегментов с атрибуцией и переходом
        к моменту в исходнике по двойному клику."""
        if not hasattr(self, 'segments_inner'):
            return
        self._update_scan_tab_title()
        for w in self.segments_inner.winfo_children():
            w.destroy()
        if not self.athlete_segments:
            tk.Label(self.segments_inner,
                     text="Сегментов нет.\nЗапустите скан атлетов\n(сохранять - выкл, атлеты - вкл)",
                     fg='#777', justify=tk.LEFT).pack(pady=8, anchor='w')
            return
        names = list(self.athlete_mapping)
        names += [n for n in self._registry_banks() if n not in names]
        # имена базовых банков доступны для привязки без ручного ввода
        names += [n for n in self._base_banks() if n not in names]
        filt = self.filter_var.get()
        for seg in sorted(self.athlete_segments,
                          key=lambda s: (s.get('file', ''), s.get('start', 0))):
            athlete = seg.get('athlete')
            if filt == 'Неизвестно' and athlete:
                continue
            if filt == 'На подтверждение' and seg.get('confirmed'):
                continue
            if filt not in ('Все', 'Неизвестно', 'На подтверждение') \
                    and athlete != filt:
                continue
            row = tk.Frame(self.segments_inner)
            row.pack(fill=tk.X, pady=1)
            seg_choices = ['— снять', '➕ Новый / поиск…'] + names
            if seg.get('confirmed'):
                chip = self._athlete_chip(
                    row, f"{athlete}", seg_choices,
                    lambda ch, s=seg: self._on_segment_select(s, ch),
                    bg='#4caf50')
            elif athlete:
                chip = self._athlete_chip(
                    row, f"{athlete}", seg_choices,
                    lambda ch, s=seg: self._on_segment_select(s, ch),
                    bg='#ff9800')
                ok = tk.Label(row, text=f"✓{seg.get('sim', 0) * 100:.0f}%",
                              bg='#4caf50', fg='white',
                              font=('Arial', 8, 'bold'), cursor='hand2',
                              padx=3, pady=1)
                ok.bind('<Button-1>',
                        lambda e, s=seg, b=athlete: self.confirm_segment(s, b))
                ok.pack(side=tk.LEFT)
            else:
                chip = self._athlete_chip(
                    row, None, seg_choices,
                    lambda ch, s=seg: self._on_segment_select(s, ch))
            chip.pack(side=tk.LEFT)
            if athlete:
                self._bind_tooltip(chip, athlete)
            t0, t1 = seg.get('start', 0.0), seg.get('end', 0.0)
            info = tk.Label(
                row, font=('Arial', 8),
                text=f"{os.path.splitext(seg.get('file', '?'))[0]}  "
                     f"{int(t0 // 60)}:{t0 % 60:04.1f}–{int(t1 // 60)}:{t1 % 60:04.1f}")
            info.pack(side=tk.LEFT, padx=(4, 0))
            for w in (row, info):
                w.bind('<Double-1>',
                       lambda e, s=seg: self.open_video(
                           s.get('source'), at_real_seconds=s.get('thumb_t')))

    def _get_attempt_sims(self, attempt):
        """Похожести попытки на банки атлетов, по убыванию."""
        try:
            store = self._load_profile_store()
            banks = self._athlete_banks(store)
            if attempt in store and banks:
                return sorted(((ath, self._attempt_athlete_sim(store[attempt], b)[0])
                               for ath, b in banks.items()), key=lambda kv: -kv[1])
        except Exception:
            pass
        return []

    def _refresh_attempt_tile(self, attempt):
        """Обновляет строку атрибуции одной плитки, не пересобирая сетку
        (скролл сохраняется)."""
        attr_frame = self._attempt_attr_frames.get(attempt)
        label = self._attempt_labels.get(attempt)
        if not attr_frame or not attr_frame.winfo_exists():
            self.update_attempt_thumbnails()
            return
        for w in attr_frame.winfo_children():
            w.destroy()
        srt = self._get_attempt_sims(attempt)
        self._build_attr_row(attr_frame, attempt, srt)
        if label is not None and label.winfo_exists():
            assigned_athlete = next((ath for ath, files in self.athlete_mapping.items()
                                     if attempt in files), None)
            tip = self._attribution_tip(attempt, assigned_athlete, srt)
            if tip:
                self._bind_tooltip(label, tip)

    def show_attempt_context_menu(self, event, attempt):
        """Контекстное меню плитки: подтвердить / привязать / снять."""
        assigned_athlete = next((ath for ath, files in self.athlete_mapping.items()
                                 if attempt in files), None)
        auto = self.attempt_assignments.get(attempt)
        menu = tk.Menu(self.root, tearoff=0)
        if auto and not assigned_athlete:
            menu.add_command(label=f"Подтвердить: {auto.get('athlete')}",
                             command=lambda a=attempt, b=auto.get('athlete'):
                             self.confirm_assignment(a, b))
        athletes = [ath for ath, _ in self._get_attempt_sims(attempt)]
        athletes += [ath for ath in self.athlete_mapping if ath not in athletes]
        for ath in athletes:
            if ath != assigned_athlete:
                menu.add_command(label=f"Привязать: {ath}",
                                 command=lambda a=attempt, b=ath:
                                 self.assign_attempt_to_athlete(a, b))
        if assigned_athlete or auto:
            menu.add_separator()
            menu.add_command(label="Снять привязку",
                             command=lambda a=attempt: self.remove_assignment(a))
        if menu.index(tk.END) is not None and menu.index(tk.END) > 0:
            menu.tk_popup(event.x_root, event.y_root)

    def display_image(self, cv_image):
        bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(bgr_image)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.image = self.photo

    def open_output_folder(self):
        folder = self.output_folder
        if os.path.exists(folder):
            if sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            elif sys.platform == "win32":
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", folder])
        else:
            messagebox.showerror("Ошибка", "Папка не найдена.")
        self.processing_config_file = os.path.join(self.output_folder, "processing.yaml")

    def open_video(self, filename, at_real_seconds=None):
        """Открывает видео в VLC/QuickTime; at_real_seconds - реальная секунда
        исходника (для сегментов скана), пересчитывается в файловое время."""
        try:
            # Преобразуем относительный путь в абсолютный
            abs_path = self.get_absolute_path(filename)
            if not os.path.exists(abs_path):
                self.log(f"Файл не найден: {abs_path}")
                return

            if at_real_seconds is not None and self._active_slowmo_factor > 1:
                t_seek = at_real_seconds * self._active_slowmo_factor
            else:
                t_seek = at_real_seconds if at_real_seconds is not None else 0.7

            if sys.platform == "darwin":  # macOS
                # Скрипт открывает файл и сразу запускает воспроизведение
                applescript = f'''
                tell application "QuickTime Player"
                    activate
                    open POSIX file "{abs_path}"
                    delay 0.5
                    tell document 1
                        try
                            set sound volume to 0.05
                        end try
                        set ts to time scale
                        set current time to (ts * {t_seek:.3f}) as integer
                        play
                    end tell
                end tell
                '''
                subprocess.Popen(["osascript", "-e", applescript], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # subprocess.run(["open", "-a", "QuickTime Player", abs_path])
                # subprocess.Popen(['open', '-a', 'VLC', abs_path])
            elif sys.platform == "win32":  # Windows
                # Пробуем найти VLC в стандартных местах установки
                vlc_paths = [
                    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
                    "vlc"  # Пробуем системный путь
                ]
                
                vlc_found = False
                for vlc_path in vlc_paths:
                    try:
                        if os.path.exists(vlc_path) or vlc_path == "vlc":
                            subprocess.Popen([vlc_path, f"--start-time={t_seek:.2f}", abs_path])
                            vlc_found = True
                            break
                    except Exception:
                        continue
                
                if not vlc_found:
                    # Если VLC не найден, пробуем открыть файл системным способом
                    os.startfile(abs_path)
            else:  # Linux
                subprocess.Popen(['vlc', f"--start-time={t_seek:.2f}", abs_path])

            self.log(f"Открыто видео: {filename}")
        except Exception as e:
            self.log(f"Ошибка при открытии видео: {str(e)}")
            # Пробуем открыть файл системным способом как запасной вариант
            try:
                os.startfile(abs_path)
            except Exception as e2:
                self.log(f"Не удалось открыть файл системным способом: {str(e2)}")

    def poll_attempts(self):
        """Проверяет появление новых попыток и обновляет UI"""
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.is_alive():
            pass  # не обновляем, если идёт обработка
        else:
            self.update_attempt_thumbnails()

        # Следующий опрос через 3 секунды
        self.root.after(3000, self.poll_attempts)

    def stop_processing(self):
        if not self.processing:
            messagebox.showinfo("Информация", "Нет активной обработки.")
            return

        self.processing = False
        self.log("Обработка прервана пользователем.")

        # Кнопки переключат финальные блоки задач после остановки всех потоков
        self.progress_var.set(0)

    def load_processing_config(self):
        """Загружает конфигурацию обработки из файла"""
        if os.path.exists(self.processing_config_file):
            with open(self.processing_config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                if 'processing-config' in config:
                    # Загружаем ROI
                    if 'roi' in config['processing-config']:
                        roi_data = config['processing-config']['roi']
                        if roi_data is not None:
                            if isinstance(roi_data, dict):
                                if 'rectangle' in roi_data:
                                    self.roi = tuple(roi_data['rectangle']) # Преобразуем обратно в tuple
                                    self.roi_mode = "rectangle"
                                elif 'polygon' in roi_data:
                                    # Убеждаемся, что каждая точка является list
                                    polygon_points = []
                                    for point in roi_data['polygon']:
                                        if isinstance(point, tuple):
                                            polygon_points.append(list(point))
                                        else:
                                            polygon_points.append(point)
                                    self.roi = polygon_points # Многоугольник в формате list
                                    self.roi_mode = "polygon"
                            # Поддержка старого формата для обратной совместимости
                            elif isinstance(roi_data, list):
                                if len(roi_data) == 4:  # Прямоугольник (две точки: x1,y1,x2,y2)
                                    self.roi = tuple(roi_data)  # Преобразуем обратно в tuple
                                    self.roi_mode = "rectangle"
                                elif len(roi_data) > 4 and all(isinstance(point, list) and len(point) == 2 for point in roi_data):  # Многоугольник (список точек)
                                    self.roi = roi_data
                                    self.roi_mode = "polygon"
                    
                    # Загружаем режим ROI
                    if 'roi_mode' in config['processing-config']:
                        self.roi_mode = config['processing-config']['roi_mode']

                    # Загружаем параметры обработки
                    if 'params' in config['processing-config']:
                        saved_params = config['processing-config']['params']
                        for key in self.processing_params:
                            if key in saved_params:
                                self.processing_params[key] = float(saved_params[key])
                        self._sync_ui_from_params()

                    # Загружаем флаг слоумо (коэффициент из нового поля,
                    # для старых конфигов - 4 при включённом чекбоксе)
                    if self.slowmo_var is not None and 'slowmo_quarter' in config['processing-config']:
                        cfg = config['processing-config']
                        factor = int(cfg.get('slowmo_factor', 4)) if cfg.get('slowmo_quarter') else 1
                        self._set_slowmo_ui(factor)

                    # Загружаем стратегию обработки (из чекбоксов)
                    if 'strategy' in config['processing-config']:
                        strategy = config['processing-config']['strategy']
                        if strategy in ('auto', 'manual', 'attempts', 'athletes'):
                            if hasattr(self, 'save_now_var') and self.save_now_var is not None:
                                self.save_now_var.set(strategy in ('auto', 'attempts'))
                                self.athletes_var.set(strategy in ('attempts', 'athletes'))

                    # Восстанавливаем выбранные файлы этого дня
                    files = config['processing-config'].get('selected_files') or []
                    self.selected_files = [f for f in files if isinstance(f, str)]
                    self._render_file_list()

    def save_processing_config(self):
        """Сохраняет конфигурацию обработки в файл"""
        self._sync_params_from_ui()

        config = {
            'processing-config': {
                'roi_mode': self.roi_mode,
                'params': dict(self.processing_params),
                'slowmo_quarter': bool(self.slowmo_var.get()) if self.slowmo_var else False,
                'slowmo_factor': self._ui_slowmo_factor() if self.slowmo_var else 1,
                'strategy': self._strategy(),
                'selected_files': [f for f in self.selected_files if isinstance(f, str)],
            }
        }
        
        # Сохраняем ROI в зависимости от режима
        if self.roi:
            if self.roi_mode == "rectangle":
                # Прямоугольник: сохраняем как list
                config['processing-config']['roi'] = {
                    'rectangle': list(self.roi)
                }
            elif self.roi_mode == "polygon":
                # Многоугольник: сохраняем как список точек
                # Убеждаемся, что каждая точка тоже является list, а не tuple
                polygon_points = []
                for point in self.roi:
                    if isinstance(point, tuple):
                        polygon_points.append(list(point))
                    else:
                        polygon_points.append(point)
                config['processing-config']['roi'] = {
                    'polygon': polygon_points
                }
        else:
            config['processing-config']['roi'] = None
        
        os.makedirs(os.path.dirname(self.processing_config_file), exist_ok=True)
        with open(self.processing_config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)


    def show_canvas_with_roi(self, canvas_path):
        """Показывает изображение canvas и сохранённый ROI поверх него"""
        if not os.path.exists(canvas_path):
            return False
        try:
            image = Image.open(canvas_path)
            self.display_image(np.array(image))
        except Exception as e:
            self.log(f"Ошибка отображения canvas: {e}")
            return False
        # Если есть сохраненный ROI, показываем его
        if self.roi:
            if self.roi_mode == "rectangle":
                x1, y1, x2, y2 = self.roi
                self.canvas.create_rectangle(
                    x1 * self.canvas_width / 100,
                    y1 * CANVAS_HEIGHT / 100,
                    x2 * self.canvas_width / 100,
                    y2 * CANVAS_HEIGHT / 100,
                    outline="red",
                    tags="roi_rectangle"
                )
            elif self.roi_mode == "polygon":
                # Конвертируем точки из процентов в пиксели
                pixel_points = [(x * self.canvas_width / 100, y * CANVAS_HEIGHT / 100) for x, y in self.roi]

                # Рисуем точки
                for x, y in pixel_points:
                    self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="roi_points")

                # Рисуем линии многоугольника
                for i in range(len(pixel_points)):
                    p1 = pixel_points[i]
                    p2 = pixel_points[(i + 1) % len(pixel_points)]
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                            fill="red", width=2, tags="roi_polygon")
        return True

    def refresh_canvas_from_folder(self):
        """Показывает canvas.jpg текущей рабочей папки (или приложения) с сохранённым ROI"""
        canvas_path = os.path.join(self.output_folder, "canvas.jpg")
        if not os.path.exists(canvas_path):
            canvas_path = os.path.join(self._app_folder, "canvas.jpg")
        return self.show_canvas_with_roi(canvas_path)

    def create_canvas_from_video(self, video_path):
        if os.path.exists(self.output_folder):
            work_folder = self.output_folder
        else:
            work_folder = self._app_folder

        canvas_path = os.path.join(work_folder, "canvas.jpg")
        """Создает canvas.jpg из видео"""
        try:
            # Получаем длительность видео
            cmd = [get_ffprobe_path(),
                   '-v',
                   'error',
                   '-show_entries',
                   'format=duration',
                   '-of',
                   'default=noprint_wrappers=1:nokey=1',
                   video_path]
            duration = float(subprocess.check_output(cmd).decode().strip())

            # Берем кадр из середины видео
            seek_time = duration / 2

            # Создаем canvas.jpg

            cmd = [
                get_ffmpeg_path(),
                '-ss', str(seek_time),
                '-i', video_path,
                '-vframes', '1',
                        '-vf', f'scale={self.canvas_width}:{CANVAS_HEIGHT}',
                '-y',
                canvas_path
            ]

            # Отображаем canvas.jpg
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                # Показываем кадр на канвасе и сохранённый ROI
                self.show_canvas_with_roi(canvas_path)
                # return canvas_path
            except subprocess.CalledProcessError as e:
                print(f"Ошибка создания canvas.jpg (1): {e}")
                return None
            return True
        except Exception as e:
            self.log(f"Ошибка создания canvas.jpg (2): {e}")
            return False

    def get_filtered_attempts(self):
        """Возвращает отфильтрованный список попыток"""
        if not os.path.exists(self.output_folder):
            return []

        files = [f for f in os.listdir(self.output_folder) if f.endswith(".mp4")]
        files = [os.path.join(self.output_folder, f) for f in files]

        # Применяем фильтр по атлетам (mapping хранит базовые имена файлов,
        # сравниваем по basename; во всех ветках возвращаем полные пути)
        if self.filter_var.get() == "Все":
            filtered_files = files
        elif self.filter_var.get() == "Неизвестно":
            assigned = {os.path.basename(a)
                        for atts in self.athlete_mapping.values() for a in atts}
            filtered_files = [f for f in files
                              if os.path.basename(f) not in assigned]
        elif self.filter_var.get() == "На подтверждение":
            # нет привязки вообще + авто-привязки (провизорные)
            assigned = {os.path.basename(a)
                        for atts in self.athlete_mapping.values() for a in atts}
            filtered_files = [f for f in files
                              if os.path.basename(f) in self.attempt_assignments
                              or os.path.basename(f) not in assigned]
        else:
            athlete = self.filter_var.get()
            # подтверждённые + неподтверждённые с предложенной авто-детекцией
            names = list(self.athlete_mapping.get(athlete, []))
            names += [n for n, a in self.attempt_assignments.items()
                      if a.get('athlete') == athlete and n not in names]
            filtered_files = [os.path.join(self.output_folder, a)
                              for a in sorted(set(names))
                              if os.path.exists(os.path.join(self.output_folder, a))]

        # Применяем фильтры по рейтингу
        if self.active_rating_filters:
            rating_filtered_files = []
            for file_path in filtered_files:
                file_name = os.path.basename(file_path)
                if file_name in self.attempt_ratings:
                    rating = self.attempt_ratings[file_name]
                    # Показываем файл, если он соответствует хотя бы одному активному фильтру
                    if any(rating.get(filter_type, False) for filter_type in self.active_rating_filters):
                        rating_filtered_files.append(file_path)
                else:
                    # Если у файла нет рейтинга, показываем его только если нет активных фильтров
                    if not self.active_rating_filters:
                        rating_filtered_files.append(file_path)
            filtered_files = rating_filtered_files

        return sorted(filtered_files)

    def on_attempt_drag_start(self, event, attempt):
        """Обработчик начала перетаскивания попытки"""
        # Сохраняем начальные координаты
        self.drag_start_x = event.x_root
        self.drag_start_y = event.y_root
        self.drag_attempt = attempt

        # Запускаем таймер для проверки, является ли это перетаскиванием
        self.drag_timer = self.root.after(200, self.check_drag)

    def check_drag(self):
        """Проверяет, является ли действие перетаскиванием"""
        if not hasattr(self, 'drag_start_x'):
            return

        # Если курсор сдвинулся достаточно далеко, начинаем перетаскивание
        if (abs(self.root.winfo_pointerx() - self.drag_start_x) > 5 or
                abs(self.root.winfo_pointery() - self.drag_start_y) > 5):
            self.start_drag()
        else:
            # Если курсор не сдвинулся, это был клик
            self.root.after_cancel(self.drag_timer)

    def start_drag(self):
        """Начинает перетаскивание"""
        if not hasattr(self, 'drag_attempt'):
            return

        # Создаем окно для перетаскивания
        self.drag_window = tk.Toplevel(self.root)
        self.drag_window.overrideredirect(True)
        self.drag_window.attributes('-alpha', 0.7)

        # Создаем метку с изображением
        thumbnail_path = os.path.join(self.output_folder, f"{os.path.splitext(self.drag_attempt)[0]}.jpg")
        if os.path.exists(thumbnail_path):
            try:
                image = Image.open(thumbnail_path)
                image = image.resize((100, 75), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                label = tk.Label(self.drag_window, image=photo)
                label.image = photo
                label.pack()
            except Exception as e:
                self.log(f"Ошибка при создании превью для перетаскивания: {str(e)}")

        # Размещаем окно под курсором
        x = self.root.winfo_pointerx() - 50
        y = self.root.winfo_pointery() - 37
        self.drag_window.geometry(f"+{x}+{y}")

        # Запускаем обновление позиции
        self.update_drag_window()

    def update_drag_window(self):
        # Реализация обновления позиции окна перетаскивания
        pass

    def show_attempt_context_menu(self, event, attempt):
        # Реализация контекстного меню для попытки
        pass




    def on_output_folder_changed(self):
        self.processing_config_file = os.path.join(self.output_folder, "processing.yaml")
        self.candidates_file = os.path.join(self.output_folder, "candidates.yaml")
        # Создаем выходную папку, если её нет
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        canvas_path_app = os.path.join(self._app_folder, "canvas.jpg")
        canvas_path_out = os.path.join(self.output_folder, "canvas.jpg")
        if os.path.exists(canvas_path_app) and not os.path.exists(canvas_path_out) :
            shutil.copy(canvas_path_app, canvas_path_out)

        self.load_processing_config()
        # Настраиваем логгер
        if not self.logger:
            self.setup_logger()

        # Загружаем кандидатов ручного режима
        self.load_candidates()
        self.update_cut_button_state()

        # Перезагружаем маппинг из новой папки
        self.load_athlete_mapping()
        self._load_assignments()
        self._load_segments()
        
        # Загружаем рейтинги
        self.load_ratings()

        # Обновляем интерфейс
        self.update_athlete_list()
        self.update_attempt_thumbnails()
        self.update_folder_status()
        self._render_file_list()
        # Показываем canvas и сохранённый ROI новой рабочей папки
        self.refresh_canvas_from_folder()



class VideoPreviewPopup:
    """Встроенный видеоплеер-попап (cv2 + ImageTk, без внешних зависимостей).
    Space/клик - пауза, стрелки - перемотка, Esc - закрыть. Времена в статусе
    и для внешнего плеера - в реальных секундах (factor - замедление записи)."""

    MAX_WIDTH = 960

    def __init__(self, app, video_path, start_file_sec, title, factor=1,
                 attempt=None):
        self.app = app
        self._path = video_path
        self.attempt = attempt
        # ключ атрибуции: клип-строка или номер AttemptInfo (NNNN.mp4)
        self.athlete_key = attempt if isinstance(attempt, str) \
            else (f"{attempt.number:04d}.mp4" if attempt is not None else None)
        self._next_key = None
        self.factor = max(1, factor)
        self.playing = True
        # скорость относительно РЕАЛЬНОГО времени: x1 = как в жизни.
        # В _tick она умножается на factor (замедление записи файла)
        self.speed = 1.0
        self._carry = 0.0  # дробный остаток продвигаемых кадров
        self._after_id = None  # id отложенного тика (для отмены)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showinfo("Превью", "Не удалось открыть видео")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        # реальная скорость контента = fps * factor кадров/с; показываем
        # максимум 25 к/с, остальные кадры проматываем grab() - та же
        # скорость движения при меньшей нагрузке на декод
        self.base_rate = fps * self.factor          # кадров/с при x1 (реал)
        self.disp_fps = min(self.base_rate, 25.0)
        self.frame_delay = 1000.0 / max(1.0, self.disp_fps)
        # длительность клипа в реальных секундах (для статуса)
        n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_real = n_frames / fps / self.factor if fps else 0.0
        # размер файла для статуса
        try:
            size = os.path.getsize(video_path)
            self._size_str = (f"{size / 1024 ** 3:.1f} ГБ" if size >= 1024 ** 3
                              else f"{size / 1024 ** 2:.0f} МБ")
        except OSError:
            self._size_str = ""
        self.top = tk.Toplevel(app.root)
        self.top.title(title)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.label = tk.Label(self.top, bg='black', cursor="hand2")
        self.label.pack()
        bar = tk.Frame(self.top)
        bar.pack(fill=tk.X)
        self.status_var = tk.StringVar()
        tk.Label(bar, textvariable=self.status_var, anchor='w').pack(side=tk.LEFT, padx=6)
        tk.Label(bar, text="Space/клик - пауза   ← → ±2с   ↑ ↓ ±10с   "
                           "⇧↑ x2 (до x16)   ⇧↓ x0.5   Esc - закрыть",
                 fg='#666666').pack(side=tk.RIGHT, padx=6)
        tk.Button(bar, text="Во внешний плеер",
                  command=self.open_external).pack(side=tk.RIGHT, padx=6, pady=3)

        # Панель управления: повтор, автопереход, выбор атлета
        ctrl = tk.Frame(self.top)
        ctrl.pack(fill=tk.X)
        self.loop_var = tk.BooleanVar(value=False)
        self.auto_next_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Повтор", variable=self.loop_var).pack(side=tk.LEFT, padx=4)
        self._next_key = app.next_attempt_after(self.athlete_key) \
            if self.athlete_key else None
        cb = tk.Checkbutton(ctrl, text="Продолжать просмотр",
                            variable=self.auto_next_var)
        if self._next_key:
            cb.pack(side=tk.LEFT, padx=4)
        else:
            cb.pack(side=tk.LEFT, padx=4)
            cb.config(state=tk.DISABLED)
        self._ctrl = ctrl
        if self.attempt is not None:
            self._rebuild_selector()
            # двухфазное удаление: 1-й клик - краснеет, 2-й - удаляет и закрывает
            self._del_armed = False
            # tk.Label, а не tk.Button: на macOS aqua Button игнорирует bg -
            # красная фаза не была бы видна
            self.del_btn = tk.Label(ctrl, text='🗑 Удалить', font=('Arial', 9),
                                    cursor='hand2', padx=6, pady=2,
                                    bg='#dddddd')
            self._del_default_bg = self.del_btn.cget('bg')
            self.del_btn.bind('<Button-1>', lambda e: self._delete_two_phase())
            self.del_btn.pack(side=tk.RIGHT, padx=6)

        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_file_sec) * 1000)
        self.top.bind('<space>', lambda e: self.toggle_play())
        self.top.bind('<Escape>', lambda e: self.close())
        self.top.bind('<Left>', lambda e: self.seek(-2))
        self.top.bind('<Right>', lambda e: self.seek(2))
        self.top.bind('<Up>', lambda e: self.seek(-10))
        self.top.bind('<Down>', lambda e: self.seek(10))
        self.top.bind('<Shift-Up>', lambda e: self.change_speed(2.0))
        self.top.bind('<Shift-Down>', lambda e: self.change_speed(0.5))
        self.top.bind('<Shift-Right>', lambda e: self._goto(1))
        self.top.bind('<Shift-Left>', lambda e: self._goto(-1))
        self.label.bind('<ButtonPress-1>', self._on_video_press)
        self.label.bind('<B1-Motion>', self._on_video_motion)
        self.label.bind('<ButtonRelease-1>', self._on_video_release)
        # плеер поверх основного окна, не теряется под ним
        self.top.attributes('-topmost', True)
        self.top.lift()
        self.top.focus_force()
        # один экземпляр плеера: новый показ закрывает предыдущий
        # (два grab_set на macOS конфликтуют и окно может пропадать);
        # close() сохраняет геометрию - поэтому старый закрываем ДО
        # позиционирования нового
        old = getattr(app, '_preview_popup', None)
        if old is not None and old is not self and old.top.winfo_exists():
            old.close()
        app._preview_popup = self
        # позиция: по центру горизонтали у верхнего края; между попытками
        # сохраняем геометрию окна (для 'Продолжать' и Shift+стрелок)
        self.top.update_idletasks()
        prev_geom = getattr(app, '_popup_geometry', None)
        if prev_geom:
            self.top.geometry(prev_geom)
        else:
            w = self.top.winfo_reqwidth()
            x = max(0, (self.top.winfo_screenwidth() - w) // 2)
            self.top.geometry(f"+{x}+80")
        self.top.grab_set()
        self._tick()

    def _tick(self):
        if not self.playing or not self.top.winfo_exists():
            return
        # сколько кадров файла пройти за тик показа: базовая скорость x speed
        v = self.base_rate * self.speed / self.disp_fps
        self._carry += v
        n = int(self._carry)
        self._carry -= n
        for _ in range(max(0, n - 1)):
            if not self.cap.grab():
                self._on_end()
                return
        ok, frame = self.cap.read()
        if not ok:
            self._on_end()
            return
        self._show(frame)
        self._after_id = self.top.after(max(1, int(self.frame_delay)), self._tick)

    def _goto(self, delta):
        """Shift+стрелки: следующая/предыдущая попытка в том же окне"""
        keys = sorted(self.app.attempt_checkboxes.keys())
        if self.athlete_key not in keys:
            return
        i = keys.index(self.athlete_key) + delta
        if not (0 <= i < len(keys)):
            return
        nxt = keys[i]
        src = self.app.get_absolute_path(nxt)
        if not os.path.exists(src):
            return
        self.attempt = self.athlete_key = nxt
        self._load_video(src, 0.0, f"Попытка {nxt}")
        self._rebuild_selector()
        if self.top.winfo_exists():
            if self.playing:
                self._tick()
            else:
                ok, frame = self.cap.read()
                if ok:
                    self._show(frame)

    def _on_video_press(self, e):
        """Начало клика/перетаскивания за видео"""
        self._drag_start = (e.x_root, e.y_root)
        self._win_pos = (self.top.winfo_x(), self.top.winfo_y())
        self._dragged = False

    def _on_video_motion(self, e):
        """Перетаскивание окна за видео (порог 4px отличает от клика)"""
        if not getattr(self, '_drag_start', None):
            return
        dx = e.x_root - self._drag_start[0]
        dy = e.y_root - self._drag_start[1]
        if abs(dx) + abs(dy) > 4:
            self._dragged = True
        if self._dragged:
            self.top.geometry(f"+{self._win_pos[0] + dx}+{self._win_pos[1] + dy}")

    def _on_video_release(self, e):
        """Без перетаскивания это клик - пауза/плей"""
        start = getattr(self, '_drag_start', None)
        self._drag_start = None
        if start and not self._dragged:
            self.toggle_play()

    def _on_end(self):
        """Конец клипа: Повтор - с начала; Продолжать - следующая попытка
        в том же окне (без пересоздания плеера)."""
        if self.loop_var.get():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            self._tick()
            return
        if self.auto_next_var.get() and self._next_key:
            nxt = self._next_key
            src = self.app.get_absolute_path(nxt)
            if not os.path.exists(src):
                self.playing = False
                self._update_status()
                return
            self.attempt = nxt
            self.athlete_key = nxt
            self._load_video(src, 0.0, f"Попытка {nxt}")
            self._rebuild_selector()
            if self.top.winfo_exists():
                self._tick()
            return
        self.playing = False
        self._update_status()

    def _rebuild_selector(self):
        """Пересборка общего селектора атлета при смене попытки"""
        if self.attempt is None:
            return
        if getattr(self, '_sel_frame', None) and self._sel_frame.winfo_exists():
            self._sel_frame.destroy()
        self._sel_frame, self._sel_refresh = self.app._athlete_selector(
            self._ctrl, self.athlete_key, self._assign,
            on_open=self._pause,
            on_confirm=self._confirm_and_refresh)
        self._sel_frame.pack(side=tk.LEFT, padx=(8, 4))

    def _confirm_and_refresh(self, key, athlete):
        """Подтверждение авто-привязки из плеера + обновление селектора"""
        self.app.confirm_assignment(key, athlete)
        self._sel_refresh()

    def _pause(self):
        """Пауза при открытии селектора атлета"""
        if self.playing:
            self.playing = False
            self._update_status()

    def _delete_two_phase(self):
        """Удаление клипа попытки: 1-й клик - армирование (красная),
        2-й (не раньше чем через 0.6с, от случайного даблклика) - удалить."""
        if not self._del_armed:
            self._del_armed = True
            self._del_armed_t = time.monotonic()
            self.del_btn.config(text='🗑 Точно удалить?', bg='#d32f2f',
                                fg='white')
            return
        if time.monotonic() - self._del_armed_t < 0.6:
            return
        self.app.delete_attempt_file(self.athlete_key)
        self.close()

    def _disarm_delete(self):
        """Сброс армированной кнопки удаления при других действиях"""
        if getattr(self, '_del_armed', False) and self.top.winfo_exists():
            self._del_armed = False
            self.del_btn.config(text='🗑 Удалить', bg=self._del_default_bg,
                                fg='black')
    def _assign(self, choice):
        """Привязка из плеера (общий селектор) + продолжение просмотра"""
        self.app._on_selector_assign(self.athlete_key, choice)
        self._sel_refresh()
        self.app.update_attempt_thumbnails()
        self.playing = True
        self._tick()

    def _load_video(self, video_path, start_file_sec=0.0, title=None):
        """Загрузка нового клипа в то же окно плеера: чекбоксы, скорость,
        геометрия и позиция окна сохраняются."""
        self._path = video_path
        self._disarm_delete()
        if getattr(self, '_after_id', None):
            try:
                self.top.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.close()
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.base_rate = fps * self.factor
        self.disp_fps = min(self.base_rate, 25.0)
        self.frame_delay = 1000.0 / max(1.0, self.disp_fps)
        n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_real = n_frames / fps / self.factor if fps else 0.0
        try:
            size = os.path.getsize(video_path)
            self._size_str = (f"{size / 1024 ** 3:.1f} ГБ" if size >= 1024 ** 3
                              else f"{size / 1024 ** 2:.0f} МБ")
        except OSError:
            self._size_str = ""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_file_sec) * 1000)
        self._carry = 0.0
        if title:
            self.top.title(title)
        self._next_key = self.app.next_attempt_after(self.athlete_key)
        self.top.lift()

    def _show(self, frame):
        h, w = frame.shape[:2]
        if w > self.MAX_WIDTH:
            frame = cv2.resize(frame, (self.MAX_WIDTH, int(h * self.MAX_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=photo)
        self.label.image = photo
        self._update_status()

    def _real_time(self):
        """Текущая позиция в реальных секундах"""
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 / self.factor

    def _update_status(self):
        state = "▶" if self.playing else "⏸"
        total = f"/{self.total_real:.1f}с" if self.total_real > 0 else ""
        size = f" · {self._size_str}" if getattr(self, '_size_str', '') else ""
        self.status_var.set(f"{state} x{self.speed:g} реал "
                            f"{self._real_time():.1f}{total}{size}")

    def change_speed(self, mult):
        """Shift+стрелки: скорость в реальном времени, 4 нажатия в каждую
        сторону (x2..x16 и x0.5..x0.0625)"""
        self.speed = min(16.0, max(0.0625, self.speed * mult))
        self._update_status()

    def toggle_play(self):
        self.playing = not self.playing
        self._disarm_delete()
        if self.playing:
            self._tick()
        else:
            self._update_status()

    def seek(self, delta_real_sec):
        """Перемотка в РЕАЛЬНЫХ секундах (стрелки), с показом кадра"""
        self._disarm_delete()
        cur = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.cap.set(cv2.CAP_PROP_POS_MSEC,
                     max(0.0, cur + delta_real_sec * self.factor * 1000))
        ok, frame = self.cap.read()
        if ok:
            self._show(frame)

    def open_external(self):
        t_real = self._real_time()
        self.close()
        # open_video сам пересчитает реальное время в файловое
        self.app.open_video(self._path, at_real_seconds=t_real)

    def close(self):
        self.playing = False
        # запоминаем геометрию окна для следующего показа
        try:
            if self.top.winfo_exists():
                self.app._popup_geometry = self.top.geometry()
        except Exception:
            pass
        try:
            if getattr(self.app, '_preview_popup', None) is self:
                self.app._preview_popup = None
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        if self.top.winfo_exists():
            self.top.destroy()


class GuiLogHandler(logging.Handler):
    """Обработчик логов для GUI"""
    def __init__(self, app):
        super().__init__()
        self.app = app

    def emit(self, record):
        msg = self.format(record)
        try:
            self.app.root.after(0, lambda: self._append_log(msg))
        except RuntimeError:
            pass

    def _append_log(self, msg):
        self.app.log_text.configure(state='normal')
        self.app.log_text.insert(tk.END, msg + '\n')

        # Ограничиваем количество строк
        lines = self.app.log_text.get('1.0', tk.END).splitlines()
        if len(lines) > MAX_LOG_LINES:
            self.app.log_text.delete('1.0', f'{len(lines) - MAX_LOG_LINES + 1}.0')

        self.app.log_text.see(tk.END)
        self.app.log_text.configure(state='disabled')

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FreestyleParserApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Ошибка запуска", f"Не удалось запустить приложение:\n{str(e)}")
