import numpy as np
import tkinter as tk
import calendar
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont
import cv2
from PIL import Image, ImageTk
import subprocess
import datetime
import threading
import yaml
import logging
import re
import shutil
from utils import *
import time
import i18n
import ui_tr

ui_tr.install()

from Components import AthleteWidget
# splitter тянет torch+ultralytics (~1-3с) - импортируем лениво в местах
# использования, чтобы окно появлялось быстрее
import sys
import pathlib
import traceback

def _load_app_version():
    """Версия из version.txt, который build.py пишет из git-тега при сборке
    (в бандле и рядом с исходниками). Fallback - 'dev'."""
    try:
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, 'version.txt')) as f:
            v = f.read().strip()
            if v:
                return v
    except Exception:
        pass
    return 'dev'


APP_VERSION = _load_app_version()

THUMB_X = 180  # (не используется: превью хранятся полным кадром, UI масштабирует)
THUMB_Y = 120

logging.basicConfig(level=logging.INFO)

CANVAS_HEIGHT = 240

MAX_LOG_LINES = 200

def _live_num_str(num):
    """Номер live-попытки в формате файла (0005)."""
    return f"{num:04d}"

# Шаг скана: выбор из фиксированных значений (качество/скорость)
SCAN_STEP_CHOICES = ['1/2', '1', '2']

def _scan_step_to_str(value):
    """Шаг скана (сек) -> строка комбобокса ('1/2', '1', '2')."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '1'
    if abs(v - 0.5) < 1e-6:
        return '1/2'
    if abs(v - 2.0) < 1e-6:
        return '2'
    return '1'

def _scan_step_from_str(text):
    """Строка комбобокса ('1/2', '1', '2') -> шаг скана в секундах."""
    text = (text or '').strip()
    if text in ('1/2', '0.5'):
        return 0.5
    if text == '2':
        return 2.0
    return 1.0

def _natural_key(path):
    """Натуральный ключ сортировки файлов: CAM2 перед CAM10."""
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]

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

    def __init__(self, parent, initial=None, marked_dates=None):
        super().__init__(parent)
        self.title(i18n.tr("Выберите день"))
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        # дни, за которые есть папки с попытками - выделяются в сетке
        self.marked = marked_dates or set()
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
        tk.Button(buttons, text=i18n.tr("Сегодня"), command=self._pick_today).pack(side=tk.LEFT)
        tk.Button(buttons, text=i18n.tr("Отмена"), command=self.destroy).pack(side=tk.RIGHT)

        self._render()
        self.grab_set()

    def _render(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.month_label.config(text=f"{i18n.tr(self.MONTHS[self.month - 1])} {self.year}")
        for i, wd in enumerate(self.WEEKDAYS):
            tk.Label(self.grid_frame, text=i18n.tr(wd), width=4).grid(row=0, column=i)
        today = datetime.date.today()
        for r, week in enumerate(calendar.monthcalendar(self.year, self.month), start=1):
            for c, day in enumerate(week):
                if not day:
                    tk.Label(self.grid_frame, width=4).grid(row=r, column=c)
                    continue
                d = datetime.date(self.year, self.month, day)
                marked = d in self.marked
                if d == today:
                    text = f"•{day}"
                elif marked:
                    text = f"{day} •"
                else:
                    text = str(day)
                btn = tk.Button(self.grid_frame, text=text, width=4,
                                command=lambda dd=d: self._pick(dd))
                if marked:
                    btn.config(font=('Arial', 10, 'bold'))
                btn.grid(row=r, column=c)

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
        self.root.title(i18n.tr("app.title", version=APP_VERSION))
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
        ui_settings = self._load_settings()
        i18n.set_language(ui_settings.get('language', 'ru'), notify=False)
        last_folder = ui_settings.get('last_output_folder')
        if last_folder and os.path.isdir(last_folder):
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
        self._ratings_lock = threading.Lock()
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
        self.filter_var = tk.StringVar(value=i18n.tr("Все"))
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
        # Режим «Просмотр без сохранения»: попытки живут в памяти, на диск
        # пишутся только превью live_NNNN.jpg; сохранение - по кнопкам в плеере
        self.live_attempts = []  # {'num','file','start','end','thumb','saved'}
        self.live_vars = {}  # num -> tk.BooleanVar чекбокса «сохранить»
        self._live_lock = threading.Lock()
        self._live_next_num = 0
        self._live_player_opened = False

        # self.on_output_folder_changed()

        # Создаем интерфейс
        self.setup_ui()
        self._setup_window_icon()
        if os.path.exists(self.output_folder):
            self.on_output_folder_changed()
        else:
            # Папки нет - явно показываем это в интерфейсе
            self.update_folder_status()
            print(f"Рабочая папка не существует: {self.output_folder} - "
                  f"выберите день (📅) или папку (Изменить)")
        # Запускаем периодическое обновление
        self.root.after(1000, self.periodic_update)
        # после отрисовки окна: пересчитать колонки сетки попыток
        # по реальной ширине (до маппинга winfo_width() == 1)
        self.root.after_idle(self._maybe_relayout_attempts)

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
        """Периодическое обновление интерфейса (таймер живёт даже при
        ошибке отрисовки - иначе панель попыток замирает до перезапуска)"""
        try:
            # детект выхода из сна: тик раз в секунду, большой скачок
            # времени означает, что машина спала (обработка прервана)
            now = time.time()
            last = getattr(self, '_last_tick_t', None)
            self._last_tick_t = now
            if last is not None and now - last > 30:
                self.log(i18n.tr("Компьютер выходил из спящего режима. "
                                 "Если обработка остановилась - запустите её "
                                 "снова: уже обработанные файлы пропускаются, "
                                 "доделается только остаток."), logging.WARNING)
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
                self._update_progress_info()
        except Exception:
            self.log(i18n.tr("Ошибка обновления интерфейса: {err}", err=traceback.format_exc()),
                     logging.ERROR)
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
            self.log(i18n.tr("Попытка {attempt} отвязана от атлета", attempt=attempt))
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
                self.log(i18n.tr("Попытка {attempt} привязана к атлету {athlete_name}",
                                 attempt=attempt, athlete_name=athlete_name))
                return

    def on_filter_change(self, event):
        """Обработчик изменения фильтра"""
        self.need_update_attempts = True  # Устанавливаем флаг обновления

    def _settings_file(self):
        return os.path.join(self._app_folder, "settings.yaml")

    def _on_language_selected(self, event=None):
        """Смена языка из шапки: применяем и запоминаем в настройках."""
        code = self._lang_by_code.get(self.lang_var.get())
        if code:
            self._apply_language(code)

    def _apply_language(self, code):
        i18n.set_language(code)
        ui = self.settings.setdefault('ui', {})
        ui['language'] = code
        self._save_settings()
        self.root.title(i18n.tr("app.title", version=APP_VERSION))
        # Рантайм-перерисовка всех запомненных виджетов и открытых окон
        ui_tr.retext(self.root)
        self.lang_var.set(code.upper())

    def _load_settings(self):
        """Настройки из ~/.FreestyleParser/settings.yaml (старый плоский формат
        с last_output_folder мигрируем)."""
        try:
            f = self._settings_file()
            if os.path.exists(f):
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                ui = data.get('ui') or {}
                self.settings = {'ui': ui}
                last = data.get('last_output_folder')
                if last:
                    ui.setdefault('last_output_folder', last)
                return ui
        except Exception:
            pass
        self.settings = {'ui': {}}
        return self.settings['ui']

    def _save_settings(self):
        """Сохраняет настройки (ui.*)."""
        try:
            with open(self._settings_file(), "w", encoding="utf-8") as f:
                yaml.dump({'ui': self.settings.get('ui', {})}, f,
                          allow_unicode=True)
        except Exception as e:
            print(f"Не удалось сохранить настройки: {e}")

    def _set_selected_files(self, files):
        """Единая точка установки списка файлов: натуральная сортировка
        (иначе порядок кликов в диалоге/порядок в yaml непредсказуем)
        + перерисовка списка."""
        self.selected_files = sorted([f for f in files if isinstance(f, str)],
                                     key=_natural_key)
        self._render_file_list()

    def _restore_selected_files(self):
        """Восстанавливает выбор файлов этого дня (processing.yaml в папке дня):
        недоделанная работа видна сразу; отсутствующие файлы - красным."""
        try:
            with open(self.processing_config_file, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            files = (config.get('processing-config') or {}).get('selected_files') or []
        except Exception:
            files = []
        self._set_selected_files(files)
        if not self.selected_files:
            self.current_video_for_roi = None
            return
        existing = [f for f in self.selected_files if os.path.exists(f)]
        if existing:
            try:
                self.current_video_for_roi = existing[0]
                self.current_frame_position = 0.5
                self.create_canvas_from_video(existing[0])
                factor = detect_slowmo_factor(existing[0])
                self._set_slowmo_ui(factor)
            except Exception as e:
                self.log(i18n.tr("Не удалось восстановить предпросмотр: {err}", err=e), logging.WARNING)
        missing = len(self.selected_files) - len(existing)
        self.log(i18n.tr("Восстановлен выбор файлов: {n}", n=len(self.selected_files))
                 + (i18n.tr(" (нет на диске: {missing} - отмечены красным)", missing=missing) if missing else ""))

    def _render_file_list(self):
        """Список выбранных файлов с отметками состояния:
        отсутствует на диске - красным с суффиксом; в папке с частичной
        работой (есть отметки/кандидаты/сегменты) необработанные файлы -
        красным. В чистой папке ничего не краснеет."""
        if not hasattr(self, 'listbox_files'):
            return
        self.listbox_files.delete(0, tk.END)
        if hasattr(self, 'files_hint_label'):
            self.files_hint_label.pack_forget() if self.selected_files \
                else self.files_hint_label.pack(fill=tk.X, padx=5)
        folder_ready = os.path.isdir(self.output_folder)
        partial_work = folder_ready and self._folder_has_work()
        processed = self._load_processed()
        for i, f in enumerate(self.selected_files):
            base = os.path.basename(f)
            exists = os.path.exists(f)
            self.listbox_files.insert(tk.END,
                                      base + ('' if exists else i18n.tr('  (нет файла)')))
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
        state = i18n.tr("завершён") if completed else i18n.tr("ПРЕРВАН")
        self.log(i18n.tr("Итог [{strategy}]: {state}, время {mins} мин {secs:02d} с, "
                 "файлов: {n}", strategy=strategy, state=state, mins=mins,
                 secs=secs, n=len(files)))

    def _is_processed(self, video_path, strategy):
        """Файл уже полностью обработан этой стратегией (отметка в
        processed.yaml ставится только после успешного прогона файла)."""
        mark = self._load_processed().get(os.path.basename(video_path))
        return bool(mark and mark.get('strategy') == strategy)

    # === Журнал нарезанных попыток (папка дня, attempts_index.yaml) ===

    def _attempts_index_file(self):
        return os.path.join(self.output_folder, 'attempts_index.yaml')

    def _load_attempts_index(self):
        """[{number, source, start, end}] - что уже нарезано в этой папке."""
        f = self._attempts_index_file()
        if os.path.exists(f):
            try:
                with open(f, encoding='utf-8') as fh:
                    data = yaml.safe_load(fh) or {}
                return data.get('attempts') or []
            except Exception:
                pass
        return []

    def _attempt_already_cut(self, source_video, start):
        """Попытка из этого исходника около времени start уже нарезана
        (защита от дублей при докатке прерванного файла: недоконченный
        файл пересканируется целиком, а его попытки уже на диске).
        Допуск 2с: границы детекции между прогонами слегка гуляют,
        а соседние попытки разделены минимум min_pause."""
        src = os.path.basename(source_video)
        for a in self._load_attempts_index():
            if a.get('source') == src and \
                    abs(float(a.get('start', 0)) - start) < 2.0:
                return True
        return False

    def _record_attempt(self, attempt):
        """Пишет попытку в attempts_index.yaml (атомарно: tmp + replace),
        сразу после успешной вырезки."""
        entries = self._load_attempts_index()
        entries.append({'number': attempt.number,
                        'source': os.path.basename(attempt.source_video),
                        'start': float(attempt.start),
                        'end': float(attempt.end)})
        try:
            tmp = self._attempts_index_file() + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                yaml.dump({'attempts': entries}, fh, allow_unicode=True)
            os.replace(tmp, self._attempts_index_file())
        except Exception as e:
            self.log(i18n.tr("Не удалось записать журнал попыток: {err}", err=e),
                     logging.WARNING)

    def _mark_processed(self, video_path, strategy):
        """Отмечает исходник обработанным (после успешного прогона файла).
        Атомарная запись: обрыв питания не оставит битый yaml."""
        data = self._load_processed()
        data[os.path.basename(video_path)] = {
            'strategy': strategy,
            'when': datetime.datetime.now().isoformat(timespec='seconds')}
        try:
            tmp = self._processed_file() + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                yaml.dump(data, fh, allow_unicode=True)
            os.replace(tmp, self._processed_file())
        except Exception as e:
            self.log(i18n.tr("Не удалось записать отметку обработки: {err}", err=e), logging.WARNING)
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
        self.log(log_message or i18n.tr("Рабочая папка: {folder}", folder=folder))

    def update_folder_status(self):
        """Показывает состояние рабочей папки: создана или нет"""
        if not hasattr(self, 'folder_status_label'):
            return
        if os.path.isdir(self.output_folder):
            self.folder_status_label.config(text=i18n.tr("✓ папка создана"), fg="#2e7d32")
            self.entry_output.config(bg='white')
        elif self.selected_files:
            # файлы уже выбраны - папка будет создана при сохранении конфига/обработке
            self.folder_status_label.config(
                text=i18n.tr("⏳ папка дня будет создана при запуске обработки"), fg="#ef6c00")
            self.entry_output.config(bg='#fff3e0')
        else:
            self.folder_status_label.config(
                text=i18n.tr("✗ папка не создана — выберите 📅 день или папку"), fg="#c62828")
            self.entry_output.config(bg='#ffdddd')

    def choose_day_folder(self):
        """Выбор папки дня через календарь: ~/Desktop/FreestyleParser/YYYY-MM-DD"""
        dialog = DatePicker(self.root, datetime.date.today(),
                            marked_dates=self._dates_with_attempts())
        self.root.wait_window(dialog)
        d = dialog.result
        if d is None:
            return
        folder = os.path.expanduser(f"~/Desktop/FreestyleParser/{d.isoformat()}")
        self._set_output_folder(folder, i18n.tr("Рабочая папка дня: {folder}", folder=folder))

    def _dates_with_attempts(self):
        """Дни (для календаря), за которые есть папки дня:
        ~/Desktop/FreestyleParser/YYYY-MM-DD. Папка может содержать что
        угодно (попытки, кандидаты, исходники) - важен сам факт."""
        base = os.path.expanduser("~/Desktop/FreestyleParser")
        dates = set()
        if not os.path.isdir(base):
            return dates
        for name in os.listdir(base):
            if not os.path.isdir(os.path.join(base, name)):
                continue
            try:
                dates.add(datetime.date.fromisoformat(name))
            except ValueError:
                continue
        return dates

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
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Сначала выберите файлы для обработки"))
            return

        if not self.roi:
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Сначала выберите область интереса"))
            return
        self.processing = True
        self._active_slowmo_factor = self._ui_slowmo_factor()
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        self.log(i18n.tr("Запускаем процессинг файлов с roi={roi}", roi=self.roi))
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
        self._progress_file_t0 = None
        self._progress_file_name = ''

        def run():
            try:
                target()
            except Exception:
                self.log(i18n.tr("Ошибка потока обработки: {err}", err=traceback.format_exc()),
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
            return i18n.tr("{n:.1f} ГБ", n=nbytes / 1024 ** 3)
        if nbytes >= 1024 ** 2:
            return i18n.tr("{n:.0f} МБ", n=nbytes / 1024 ** 2)
        return i18n.tr("{n:.0f} КБ", n=nbytes / 1024)

    def _remux_clip(self, source_video, start_real, duration_real, output_file,
                    slowmo, label):
        """Вырезка интервала из исходника в output_file (ffmpeg stream copy;
        для слоумо - сжатие таймстампов, см. on_attempt_file_created).
        Возвращает True при успехе."""
        ffmpeg = get_ffmpeg_path()
        if not ffmpeg:
            self.log(i18n.tr("ffmpeg не найден: ни в PATH, ни в папке bin рядом с программой"),
                     logging.ERROR)
            return False
        if slowmo > 1:
            # Слоумо: кадры в контейнере идут с интервалом 1/24с, а поток H.264
            # содержит 96 уникальных кадров в секунду. Сжимаем таймстампы в
            # slowmo раз (-itsscale) и выставляем timescale = 96000, чтобы
            # плеер воспроизводил попытку на реальной скорости (96 к/с)
            # без перекодирования. -ss/-t задаются в реальных секундах.
            ts = int(slowmo * 24) * 1000  # 96000 для 96 fps
            self.log(i18n.tr("Слоумо 1/{slowmo}: remux {start:.2f}s.."
                             "{end:.2f}s (реальных), stream copy "
                             "(itsscale={itsscale}, timescale={ts})",
                             slowmo=slowmo, start=start_real,
                             end=start_real + duration_real,
                             itsscale=1 / slowmo, ts=ts))
            cmd = [ffmpeg,
                   "-itsscale", f"{1 / slowmo}",
                   "-i", source_video,
                   "-ss", f"{start_real:.3f}",
                   "-t", f"{duration_real:.3f}",
                   "-c", "copy",
                   "-video_track_timescale", str(ts),
                   "-map", "0:v:0",
                   "-movflags", "+faststart",
                   "-y", output_file]
            if self._video_has_audio(source_video):
                cmd += ["-map", "0:a:0?"]
        else:
            cmd = [ffmpeg, "-ss", str(start_real), "-i", source_video,
                   "-t", str(duration_real), "-c", "copy", "-y", output_file]
        try:
            kwargs = {'capture_output': True, 'text': True, 'errors': 'replace',
                      'timeout': 300}
            if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(cmd, **kwargs)
            if result.returncode != 0:
                self.log(i18n.tr("Ошибка ffmpeg для {label}: {err}", label=label,
                                 err=result.stderr[-500:]), logging.ERROR)
                return False
        except subprocess.TimeoutExpired:
            self.log(i18n.tr("Таймаут ffmpeg для {label} (>300с)", label=label), logging.ERROR)
            return False
        except Exception as e:
            self.log(i18n.tr("Ошибка запуска ffmpeg для {label}: {err}", label=label, err=e), logging.ERROR)
            return False

        # Контрольная проверка: длительность выхода должна совпадать с ожидаемой.
        # Расхождение означает, что вырезка попала не туда (например, неверный
        # режим слоумо или кривые таймстампы исходника).
        try:
            out_duration = float(subprocess.check_output(
                [get_ffprobe_path(), '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', output_file]).decode().strip())
            if abs(out_duration - duration_real) > max(1.0, duration_real * 0.2):
                self.log(i18n.tr("ВНИМАНИЕ: {label}: длительность результата "
                                 "{out:.2f}s сильно отличается от ожидаемой "
                                 "{expect:.2f}s - проверьте режим слоумо",
                                 label=label, out=out_duration, expect=duration_real),
                         logging.WARNING)
        except Exception:
            pass
        return True

    def on_attempt_file_created(self, attempt):
        ## TODO analyze best_frame
        if self._attempt_already_cut(attempt.source_video, attempt.start):
            # докатка прерванного прогона: эта попытка уже нарезана
            self.log(i18n.tr("Попытка {attempt} ({src} ~{start:.0f}с) уже нарезана - пропускаю",
                             attempt=attempt.number, src=os.path.basename(attempt.source_video),
                             start=attempt.start))
            return self.processing
        imwrite_unicode(os.path.join(self.output_folder, f"{attempt.number:04d}.jpg"), attempt.best_frame)
        # cv2.imwrite(os.path.join(self.output_folder, f"{attempt.number:04d}-base.jpg"), attempt.base_frame)
        # person_roi_frame = extract_athlete_difference(attempt.base_frame, attempt.person_frame)
        # colors = self.get_colors_from_athlete(attempt.base_frame, attempt.person_frame, None, 4, 8)
        # pers_file_path = os.path.join(self.output_folder, f"{attempt.number:04d}-person-{colors}.jpg")
        # cv2.imwrite(pers_file_path, attempt.person_frame)
        # self.log(f"Colors {colors}")
        output_file = os.path.join(self.output_folder, f"{attempt.number:04d}.mp4")
        duration = attempt.end - attempt.start
        if duration <= 0:
            self.log(i18n.tr("Пропускаем попытку {attempt}: start={start:.2f}s >= end={end:.2f}s",
                             attempt=attempt.number, start=attempt.start, end=attempt.end))
            return self.processing
        slowmo = self._active_slowmo_factor
        if not self._remux_clip(attempt.source_video, attempt.start, duration,
                                output_file, slowmo,
                                i18n.tr("попытка {attempt:04d}", attempt=attempt.number)):
            return self.processing
        self._record_attempt(attempt)

        self.need_update_attempts = True
        self.log(i18n.tr("Видео попытки готово: {file}", file=output_file))
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
        """Обрабатывает видео в отдельном потоке (авто-режим: нарезать всё).
        Повторный запуск после прерывания (сон, кнопка Стоп) продолжает:
        файлы с отметкой processed.yaml пропускаются."""
        t0 = time.time()
        try:
            from splitter import process_video
            files = [f for f in self.selected_files
                     if not self._is_processed(f, 'auto')]
            skipped = len(self.selected_files) - len(files)
            if skipped:
                self.log(i18n.tr("{n} файл(ов) уже обработаны - пропускаю, "
                                 "доделывается остаток", n=skipped))
            if not files:
                self.log(i18n.tr("Все выбранные файлы уже обработаны - "
                                 "нечего делать (отметки в processed.yaml)"))
                return
            next_attempt_number = get_next_attempt_number(self.output_folder)
            process_video(files, self.roi, lambda attempt: self.on_attempt_file_created(attempt),
                          next_attempt_number, params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=self._active_slowmo_factor,
                          log_fn=self.log)
            self.log(i18n.tr("Файлы {files} успешно обработаны.", files=files))
            if self.processing:
                for f in files:
                    self._mark_processed(f, 'auto')
            self._log_processing_summary(i18n.tr("Нарезка всех попыток (авто)"),
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.log(i18n.tr("Ошибка обработки: {err} {trace}", err=e, trace=error_trace))
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
        кандидатов (кнопка «Нарезать»)."""
        self.scan_running = True
        t0 = time.time()
        try:
            from splitter import process_video
            existing = list(self.candidates)
            if existing:
                self.log(i18n.tr("Пересканирование: {n} существующих кандидатов сохраняются, "
                                 "новые добавляются без дублей", n=len(existing)))
            process_video(self.selected_files, self.roi, self.on_candidate_found,
                          1, params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=self._active_slowmo_factor,
                          scan_mode=True,
                          log_fn=self.log)
            self.save_candidates()
            if self.processing:
                for f in self.selected_files:
                    self._mark_processed(f, 'manual')
            self._log_processing_summary(i18n.tr("Скан кандидатов (ручной)"),
                                         self.selected_files, t0, self.processing)
            pending = [c for c in self.candidates if c['status'] == 'pending']
            self.log(i18n.tr("Сканирование завершено: всего кандидатов {n}. "
                             "Снимите галочки с ненужных попыток и нажмите «Нарезать».", n=len(pending)))
        except Exception as e:
            self.log(i18n.tr("Ошибка сканирования: {err} {trace}", err=e,
                             trace=traceback.format_exc()), logging.ERROR)
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
            self.button_live.config(state=tk.NORMAL)
            self.button_stop.config(state=tk.DISABLED)
            self._progress_file_t0 = None
            if hasattr(self, 'progress_info_var'):
                self.progress_info_var.set('')
        self.update_cut_button_state()

    # === Режим «Просмотр без сохранения» ===

    def start_live_preview(self):
        """Запуск скана без сохранения: попытки живут в памяти, просмотр
        в плеере, сохранение - по кнопкам/хоткеям плеера."""
        self._sync_params_from_ui()
        self.save_processing_config()
        if not self.selected_files:
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Сначала выберите файлы для обработки"))
            return
        if not self.roi:
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Сначала выберите область интереса"))
            return
        with self._live_lock:
            self.live_attempts = []
        self._live_player_opened = False
        self._live_next_num = self._next_attempt_number()
        self.processing = True
        self._active_slowmo_factor = self._ui_slowmo_factor()
        # «Определять атлетов» ON -> точный сплиттер (attempts_v2, тяжёлый),
        # OFF -> грубый скан сэмплами раз в scan_interval (splitter.py, быстрый).
        # Эмбеддинги в live-режиме всё равно никуда не идут (атрибуции тут нет),
        # так что OFF - предпочтительный вариант для быстрого просмотра
        coarse = not (self.athletes_var and self.athletes_var.get())
        self.button_process.config(state=tk.DISABLED)
        self.button_live.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        self.log(i18n.tr("Просмотр без сохранения ({mode}): "
                         "сканирую {n} файл(ов), "
                         "нумерация попыток с {start}",
                         mode=(i18n.tr("грубый скан") if coarse else i18n.tr("точный сплиттер")),
                         n=len(self.selected_files),
                         start=_live_num_str(self._live_next_num)))
        self._start_worker(self._live_scan_worker_coarse if coarse
                           else self._live_scan_worker)

    def _live_scan_worker_coarse(self):
        """Грубый скан (галочка «Определять атлетов» выключена).
        Тот же движок, что ручной скан кандидатов: сэмплы раз в scan_interval
        секунд, YOLO с пониженным порогом, без эмбеддингов и точной доводки.
        Найденные интервалы регистрируются как live-попытки."""
        t0 = time.time()
        try:
            from splitter import process_video
            factor = self._active_slowmo_factor

            def on_found(attempt):
                return self._live_found_coarse(attempt, factor)

            process_video(self.selected_files, self.roi, on_found, 1,
                          params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=factor,
                          scan_mode=True,
                          log_fn=self.log)
            # NB: файлы НЕ помечаем processed - ничего не сохранено
            with self._live_lock:
                n_total = len(self.live_attempts)
            self.log(i18n.tr("Грубый скан завершён за {t:.1f}с: "
                             "найдено попыток {n} (интервалы неточные ±{interval:.1f}с)",
                             t=time.time() - t0, n=n_total,
                             interval=self.processing_params.get('scan_interval', 2.5)))
        except Exception as e:
            self.log(i18n.tr("Ошибка грубого скана просмотра: {err} {trace}", err=e,
                             trace=traceback.format_exc()), logging.ERROR)
        finally:
            self.processing = False
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_attempt_thumbnails)

    def _live_found_coarse(self, attempt, factor):
        """Callback грубого скана: AttemptInfo -> live-попытка.
        Возвращает should_continue."""
        if not self.processing:
            return False
        if attempt.source_video not in self.selected_files:
            return self.processing
        with self._live_lock:
            num = self._live_next_num
            self._live_next_num += 1
            la = {'num': num,
                  'file': attempt.source_video,
                  'start': float(attempt.start),
                  'end': float(attempt.end),
                  'thumb_t': float(attempt.start),
                  'thumb': None,
                  'saved': False}
            thumb_path = os.path.join(self.output_folder, f"live_{num:04d}.jpg")
            la['thumb'] = thumb_path
            self.live_attempts.append(la)
        frame = getattr(attempt, 'best_frame', None)
        if frame is None:
            src = open_capture(attempt.source_video)
            src.set(cv2.CAP_PROP_POS_MSEC, float(attempt.start) * factor * 1000)
            ok, frame = src.read()
            src.release()
            if not ok:
                frame = np.zeros((360, 640, 3), np.uint8)
        if frame is not None:
            try:
                imwrite_unicode(thumb_path, frame)
            except Exception as e:
                self.log(i18n.tr("Не удалось записать превью {file}: {err}", file=thumb_path, err=e), logging.WARNING)
                la['thumb'] = None
        self.need_update_attempts = True
        self.log(i18n.tr("  Попытка {attempt}: {file} "
                         "{start:.1f}с..{end:.1f}с (грубый скан, не сохранена)",
                         attempt=_live_num_str(num),
                         file=os.path.basename(attempt.source_video),
                         start=la['start'], end=la['end']))
        self.root.after(0, self._maybe_open_live_player)
        return self.processing

    def _live_scan_worker(self):
        """Скан выбранных файлов сплиттером попыток БЕЗ нарезки.
        Найденные попытки: thumb live_NNNN.jpg + запись в self.live_attempts."""
        t0 = time.time()
        try:
            import attempts_v2
            roi_bbox = self._roi_bbox_percentages()
            start_pad = float(self.processing_params.get('attempt_start_padding', 2))
            end_pad = float(self.processing_params.get('attempt_end_padding', 0.5))
            factor = self._active_slowmo_factor
            for file_index, video_path in enumerate(self.selected_files):
                if not self.processing:
                    break
                base = os.path.splitext(os.path.basename(video_path))[0]
                self.log(i18n.tr("[{i}/{n}] "
                                 "Поиск попыток: {file} ...", i=file_index + 1,
                                 n=len(self.selected_files),
                                 file=os.path.basename(video_path)))
                tmp_dir = os.path.join(self.output_folder, f".attempts_tmp_{base}")

                def on_found(r, src_path=video_path):
                    return self._live_found_attempt(src_path, r, factor)

                try:
                    pstats = {}
                    attempts_v2.process_video(
                        video_path, tmp_dir, roi=roi_bbox,
                        should_continue=lambda: self.processing,
                        stats_out=pstats,
                        slowmo_factor=factor,
                        min_pause=float(self.processing_params.get('min_pause_duration', 2.5)),
                        min_attempt_duration=float(self.processing_params.get('min_attempt_duration', 3.0)),
                        reid_embed_rate=float(self.processing_params.get('reid_embed_rate', 5)),
                        attempt_found_cb=on_found,
                        log_cb=self.log,
                        progress_cb=lambda fi, tf, fi_=file_index, nf=len(self.selected_files):
                            self.on_processing_progress(fi_, nf, fi, tf))
                except Exception as e:
                    self.log(i18n.tr("Ошибка скана {file}: {err}", file=base, err=e), logging.ERROR)
                    continue
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            # NB: файлы НЕ помечаем processed - ничего не сохранено,
            # пользователь может потом нарезать их обычным режимом
            with self._live_lock:
                n_total = len(self.live_attempts)
            self.log(i18n.tr("Просмотр без сохранения завершён за {t:.1f}с: "
                             "найдено попыток {n} (сохранено пока не было)",
                             t=time.time() - t0, n=n_total))
        except Exception as e:
            self.log(i18n.tr("Ошибка режима просмотра: {err} {trace}", err=e,
                             trace=traceback.format_exc()), logging.ERROR)
        finally:
            self.processing = False
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_attempt_thumbnails)

    def _live_found_attempt(self, src_path, r, factor):
        """Callback сплиттера: регистрируем попытку в памяти + thumb.
        Возвращает should_continue."""
        if not self.processing:
            return False
        start_pad = float(self.processing_params.get('attempt_start_padding', 2))
        thumb_t = r.get('thumb_t') or r['start']
        # thumb: кадр, где атлет ближе всего к центру ROI (для плитки панели)
        src = open_capture(src_path)
        src.set(cv2.CAP_PROP_POS_MSEC, thumb_t * factor * 1000)
        ok, best_frame = src.read()
        src.release()
        if not ok:
            best_frame = np.zeros((360, 640, 3), np.uint8)
        with self._live_lock:
            num = self._live_next_num
            self._live_next_num += 1
            la = {
                'num': num,
                'file': src_path,
                'start': float(r['start']),
                'end': float(r['end']),
                'thumb_t': float(thumb_t),
                'thumb': os.path.join(self.output_folder, f"live_{num:04d}.jpg"),
                'saved': False,
            }
            self.live_attempts.append(la)
        imwrite_unicode(la['thumb'], best_frame)
        self.need_update_attempts = True
        self.log(i18n.tr("  Попытка {attempt}: {start:.1f}-{end:.1f}с "
                         "(не сохранена, смотрим в плеере)",
                         attempt=_live_num_str(num), start=r['start'], end=r['end']))
        self.root.after(0, self._maybe_open_live_player)
        return self.processing

    def _maybe_open_live_player(self):
        """Автооткрытие плеера на первой найденной попытке (один раз за скан)."""
        if self._live_player_opened:
            return
        with self._live_lock:
            if not self.live_attempts:
                return
            self._live_player_opened = True
        try:
            self.open_live_attempt(0)
        except Exception as e:
            self.log(i18n.tr("Не удалось открыть плеер: {err}", err=e), logging.ERROR)

    def open_live_attempt(self, idx):
        """Открыть плеер на попытке live_attempts[idx] (по двойному клику тоже)."""
        with self._live_lock:
            if not (0 <= idx < len(self.live_attempts)):
                return
            la = self.live_attempts[idx]
        view_start = max(0.0, la['start'] - float(self.processing_params.get('attempt_start_padding', 2)))
        title = (i18n.tr("Попытка {attempt} · клип {start:.0f}–{end:.0f}с "
                         "— просмотр без сохранения",
                         attempt=_live_num_str(la['num']), start=la['start'], end=la['end']))
        # плеер сам пересчитает границы через _load_live_entry; здесь только файл
        self._preview_popup = VideoPreviewPopup(
            self, la['file'], view_start * self._active_slowmo_factor,
            title, factor=self._active_slowmo_factor, attempt=None,
            live=True, live_index=idx)

    def save_live_attempt(self, la):
        """Сохранение live-попытки на диск (из кнопок/хоткеев плеера). True - успех."""
        with self._live_lock:
            if la.get('saved'):
                return True
        src = la['file']
        if not os.path.exists(src):
            self.log(i18n.tr("Исходник не найден: {file}", file=src), logging.ERROR)
            return False
        num = la['num']
        start_pad = float(self.processing_params.get('attempt_start_padding', 2))
        end_pad = float(self.processing_params.get('attempt_end_padding', 0.5))
        preroll = max(0.0, la['start'] - start_pad)
        duration = la['end'] - preroll + end_pad
        output_file = os.path.join(self.output_folder, f"{num:04d}.mp4")
        if os.path.exists(output_file):
            self.log(i18n.tr("{attempt} уже есть на диске - пропускаю", attempt=_live_num_str(num)))
            la['saved'] = True
            self.need_update_attempts = True
            return True
        slowmo = self._active_slowmo_factor
        if not self._remux_clip(src, preroll, duration, output_file, slowmo,
                                i18n.tr("попытка {attempt} (из плеера)", attempt=_live_num_str(num))):
            return False
        # jpg: live-превью уже снят с нужного кадра - копируем
        jpg_out = os.path.join(self.output_folder, f"{num:04d}.jpg")
        try:
            if la['thumb'] and os.path.exists(la['thumb']):
                shutil.copyfile(la['thumb'], jpg_out)
            else:
                cap = open_capture(src)
                cap.set(cv2.CAP_PROP_POS_MSEC, (la.get('thumb_t') or la['start']) * slowmo * 1000)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    imwrite_unicode(jpg_out, frame)
        except Exception as e:
            self.log(i18n.tr("Не удалось сохранить превью {file}: {err}", file=f"{num:04d}.jpg", err=e), logging.WARNING)
        la['saved'] = True
        self.need_update_attempts = True
        self.log(i18n.tr("Попытка {attempt} сохранена: {file}",
                         attempt=_live_num_str(num), file=output_file))
        return True

    # === Режим 'Попытки (атриб.)': сплиттер попыток + автоатрибуция ===

    ATTEMPT_SIM_THRESHOLD = 0.88   # минимальная похожесть для автоатрибуции
    ATTEMPT_SIM_MARGIN = 0.03      # отрыв от второго кандидата
    W_REID, W_BOAT = 0.7, 0.3      # вклад OSNet и лодки в похожесть

    def _roi_bbox_percentages(self):
        """BBox ROI в процентах (L, T, R, B) из прямоугольника или полигона."""
        if self._is_flat_rectangle(self.roi):
            return tuple(self.roi)
        xs = [p[0] for p in self.roi]
        ys = [p[1] for p in self.roi]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _is_flat_rectangle(roi):
        """Прямоугольник - 4 скаляра; полигон из 4 точек тоже даёт len==4."""
        return len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi)

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
                self.log(i18n.tr("Удаляю {n} устаревших записей из store: {names}",
                                 n=len(stale), names=', '.join(stale)))
                for n in stale:
                    del store[n]
            # мёртвые привязки маппинга - до построения банков и нумерации,
            # иначе новая попытка с тем же номером унаследует чужую привязку
            self._prune_athlete_mapping()
            banks = self._athlete_banks(store)
            if banks:
                self.log(i18n.tr("Банки атлетов: {names}", names=', '.join(banks.keys())))
            else:
                self.log(i18n.tr("Банков атлетов пока нет - попытки будут нарезаны без атрибуции "
                                 "(после привязки попыток к атлетам атрибуция заработает)"))
            number = self._next_attempt_number()
            start_pad = float(self.processing_params.get('attempt_start_padding', 2))
            # запас на вход атлета в ROI, как в точном анализе старого пайплайна
            margin = float(self.processing_params.get('scan_interval', 2.5)) + 1.0
            end_pad = float(self.processing_params.get('attempt_end_padding', 0.5))
            found_in_file = [0]

            def on_attempt_found(r):
                """Прогрессивная нарезка: попытка режется сразу после закрытия."""
                nonlocal number
                # докатка прерванного прогона: не расходуем номер на дубль
                if self._attempt_already_cut(r['source'], r['start']):
                    self.log(i18n.tr("Попытка ({src} ~{start:.0f}с) уже нарезана - пропускаю",
                                     src=os.path.basename(r['source']), start=r['start']))
                    return
                # preroll = start_pad (2с до входа) — даёт каякеру появиться во 2й секунде
                preroll = max(0.0, r['start'] - start_pad)
                # thumbnail: кадр, где атлет ближе всего к центру ROI
                # (attempts_v2 считает thumb_t по дистанции bbox-центра)
                thumb_t = r.get('thumb_t') or r['start']
                src = open_capture(r['source'])
                src.set(cv2.CAP_PROP_POS_MSEC, thumb_t * self._active_slowmo_factor * 1000)
                ok, best_frame = src.read()
                src.release()
                if not ok:
                    best_frame = np.zeros((360, 640, 3), np.uint8)
                from splitter import AttemptInfo
                info = AttemptInfo(r['source'], preroll,
                                   r['end'] + end_pad, number,
                                   best_frame, None, None, None)
                self.on_attempt_file_created(info)
                out_name = f"{number:04d}.mp4"
                number += 1
                found_in_file[0] += 1
                self.log(i18n.tr("  Попытка {attempt}: {start:.1f}-{end:.1f}с "
                                 "(лодка {boat}°)",
                                 attempt=out_name, start=r['start'], end=r['end'],
                                 boat=r['boat_top'] * 10 if r['boat_top'] is not None else '-'))
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
                if self._is_processed(video_path, 'attempts'):
                    # продолжение прерванного прогона: файл уже нарезан
                    self.log(i18n.tr("[{i}/{n}] {file}: уже обработан - пропускаю",
                                     i=file_index + 1, n=len(self.selected_files),
                                     file=os.path.basename(video_path)))
                    continue
                self.log(i18n.tr("[{i}/{n}] "
                                 "Поиск попыток: {file} ...", i=file_index + 1,
                                 n=len(self.selected_files),
                                 file=os.path.basename(video_path)))
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
                    self.log(i18n.tr("Ошибка обработки {file}: {err}", file=base, err=e), logging.ERROR)
                    continue
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if self.processing:
                    self._mark_processed(video_path, 'attempts')

                self._save_profile_store(store)
                if not results and self.processing:
                    pr = pstats.get('presence_ratio', 0.0)
                    self.log(i18n.tr("  Попыток не найдено. Главный атлет присутствовал в ROI "
                                     "в {pr:.0f}% времени. Если попытки в файле точно есть - "
                                     "проверьте ROI (атлет должен быть целиком внутри области) "
                                     "и пороги (мин. длительность попытки)", pr=pr * 100),
                             logging.WARNING)

                self._save_profile_store(store)

            # автоатрибуция новых попыток
            self._auto_assign_attempts(store, banks)
            self._log_processing_summary(i18n.tr("Попытки (атрибуция)"),
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            self.log(i18n.tr("Ошибка режима попыток: {err} {trace}", err=e,
                             trace=traceback.format_exc()), logging.ERROR)
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
            self.log(i18n.tr("  Автоатрибуция: {name} → {best} (sim={sim:.2f}) "
                             "- подтвердите на сетке превью",
                             name=name, best=best, sim=best_sim))
            return True
        hint = ', '.join(f'{a}={s:.2f}' for a, s in srt[:2])
        self.log(i18n.tr("  {name}: требует подтверждения ({hint})", name=name, hint=hint))
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
            self.log(i18n.tr("Автоатрибуция: предложено {n} из {total}", n=n_auto, total=len(fresh)))

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
            self.log(i18n.tr("Ошибка пересчёта автоатрибуции: {err}", err=e), logging.ERROR)

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
                self.log(i18n.tr("Банки атлетов: {names}", names=', '.join(banks.keys())))
            else:
                self.log(i18n.tr("Банков нет - сегменты будут 'не разобрано'; "
                                 "подтвердите атлета на строке сегмента - банк пополнится"))
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
                    who = i18n.tr("не разобрано")
                self.log(i18n.tr("  Сегмент: {who} — {start:.1f}–{end:.1f}с "
                                 "({file})", who=who, start=seg['start'],
                                 end=seg['end'], file=base))
                # живое обновление панели: сегменты появляются по мере скана
                self.athlete_segments = list(segs)
                self.need_update_segments = True

            for file_index, video_path in enumerate(self.selected_files):
                if not self.processing:
                    break
                if self._is_processed(video_path, 'athletes'):
                    # продолжение прерванного прогона: сегменты файла уже есть
                    self.log(i18n.tr("[{i}/{n}] {file}: уже обработан - пропускаю",
                                     i=file_index + 1, n=len(self.selected_files),
                                     file=os.path.basename(video_path)))
                    continue
                base = os.path.splitext(os.path.basename(video_path))[0]
                self.log(i18n.tr("[{i}/{n}] "
                                 "Скан атлетов: {file} ...", i=file_index + 1,
                                 n=len(self.selected_files),
                                 file=os.path.basename(video_path)))
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
                    self.log(i18n.tr("Ошибка обработки {file}: {err}", file=base, err=e), logging.ERROR)
                    continue
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if self.processing:
                    self._mark_processed(video_path, 'athletes')

            self.athlete_segments = segs
            self._save_segments()
            self._save_segment_profiles()
            self.log(i18n.tr("Скан атлетов завершён: {n} сегментов. "
                             "Подтверждайте атлетов на вкладке 'Скан'", n=len(segs)))
            self._log_processing_summary(i18n.tr("Скан атлетов"),
                                         self.selected_files, t0, self.processing)
        except Exception as e:
            self.log(i18n.tr("Ошибка скана атлетов: {err} {trace}", err=e,
                             trace=traceback.format_exc()), logging.ERROR)
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
            self.log(i18n.tr("Сегмент подтверждён: {athlete} — {start:.1f}–"
                             "{end:.1f}с ({file}) - банк пополнен",
                             athlete=athlete, start=seg['start'],
                             end=seg['end'], file=seg['file']))
        else:
            self.log(i18n.tr("Сегмент подтверждён: {athlete} ({file})",
                             athlete=athlete, file=seg['file']))
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
                imwrite_unicode(thumb_path, attempt.best_frame)
            except Exception:
                thumb_path = None
            self.candidates.append({
                'file': attempt.source_video,
                'start': float(attempt.start),
                'end': float(attempt.end),
                'thumbnail': thumb_path,
                'num': idx,
                'selected': True,
                'status': 'pending',
            })
        self.need_update_attempts = True
        self.log(i18n.tr("Кандидат #{idx}: {file} "
                         "{start:.1f}s..{end:.1f}s", idx=idx,
                         file=os.path.basename(attempt.source_video),
                         start=attempt.start, end=attempt.end))
        return self.processing

    def _candidate_key(self, cand):
        return (cand['file'], round(cand['start'], 2))

    def _candidate_num(self, cand):
        """Номер кандидата для показа (#003): из cand['num'], для старых
        записей candidates.yaml - из имени превью cand_NNN.jpg."""
        num = int(cand.get('num') or 0)
        if not num:
            m = re.search(r'cand_(\d+)',
                          os.path.basename(cand.get('thumbnail') or ''))
            num = int(m.group(1)) if m else 0
        return num

    def _candidate_title(self, cand):
        """Заголовок окна плеера кандидата: исходник, номер, диапазон, длительность"""
        stem = os.path.splitext(os.path.basename(cand['file']))[0]
        dur = max(0.0, float(cand['end']) - float(cand['start']))
        return (f"{stem} · #{self._candidate_num(cand):03d} · "
                f"{cand['start']:.0f}–{cand['end']:.0f}s · {dur:.0f}с")

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
        # компактный текст: панель попыток узкая, длинная подпись обрезается
        text = i18n.tr("✂️ Нарезать")
        if n > 0:
            try:
                start_pad = self.processing_params['attempt_start_padding']
                end_pad = self.processing_params['attempt_end_padding']
                scan_interval = float(self.processing_params.get('scan_interval', 2.5))
                margin = scan_interval + 1.0  # как в _cut_candidates_thread
                total = sum(self._estimate_output_bytes(
                    c['file'], c['end'] - c['start'] + start_pad + end_pad + margin)
                    for c in selected)
                text = i18n.tr("✂️ Нарезать ({n}) ≈{size}", n=n,
                               size=self._fmt_size(total)) if total > 0 \
                    else i18n.tr("✂️ Нарезать ({n})", n=n)
            except Exception:
                text = i18n.tr("✂️ Нарезать ({n})", n=n)
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
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Не выбрано ни одного кандидата"))
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
        self.log(i18n.tr("Нарезка {n} кандидатов, слоумо "
                         "{slowmo}", n=len(selected),
                         slowmo=(i18n.tr("1/{f}", f=self._active_slowmo_factor)
                                 if self._active_slowmo_factor > 1 else i18n.tr("нет"))))
        threading.Thread(target=self._cut_candidates_thread, args=(selected,), daemon=True).start()

    def _cut_candidates_thread(self, selected):
        """Точная стадия ручного режима: для каждого кандидата - полный анализ
        в узком окне вокруг него и вырезка найденных попыток."""
        t0 = time.time()
        try:
            from splitter import load_yolo_model
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
                    self.log(i18n.tr("Нарезка остановлена: обработано {done} из {total}, "
                                     "осталось {remaining} кандидатов. Повторный запуск "
                                     "«Нарезать» продолжит с того же места.",
                                     done=i, total=total, remaining=remaining))
                    break
                left = total - i
                if any(f == cand['file'] and max(s, cand['start']) < min(e, cand['end'])
                       for f, s, e in cut_windows):
                    self.log(i18n.tr("[{i}/{total}, осталось {left}] Кандидат "
                                     "{file} {start:.1f}s..{end:.1f}s "
                                     "пересекается с уже нарезанной попыткой - пропущен",
                                     i=i + 1, total=total, left=left - 1,
                                     file=os.path.basename(cand['file']),
                                     start=cand['start'], end=cand['end']))
                    continue
                # влево ищем точное начало: margin по умолчанию (шаг скана + 1)
                # или явный 'precise_left_margin' (кандидат может опаздывать)
                left_margin = float(self.processing_params.get('precise_left_margin', 0)
                                    or 0) or margin
                win_start = max(0.0, cand['start'] - start_pad - left_margin)
                win_end = cand['end'] + end_pad + margin
                next_num = get_next_attempt_number(self.output_folder)
                self.log(i18n.tr("[{i}/{total}, осталось {left}] Точный анализ "
                                 "{file} {start:.1f}s..{end:.1f}s",
                                 i=i + 1, total=total, left=left - 1,
                                 file=os.path.basename(cand['file']),
                                 start=cand['start'], end=cand['end']))
                found = []

                def fine_callback(attempt, found=found):
                    found.append(attempt)
                    return self.on_attempt_file_created(attempt)

                try:
                    from splitter import process_video
                    process_video([cand['file']], self.roi, fine_callback, next_num,
                                  params=self.processing_params,
                                  should_continue=lambda: self.processing,
                                  slowmo_factor=slowmo,
                                  model=model,
                                  restrict_range=(win_start, win_end),
                                  log_fn=self.log)
                except Exception as e:
                    self.log(i18n.tr("Ошибка точного анализа: {err} {trace}", err=e,
                                     trace=traceback.format_exc()), logging.ERROR)
                if found:
                    cand['status'] = 'cut'
                    cut_count += len(found)
                    cut_windows.append((cand['file'], cand['start'], cand['end']))
                else:
                    self.log(i18n.tr("В окне кандидата попыток не найдено - пропущен"))
                self.save_candidates()
            if not stopped:
                self.log(i18n.tr("Нарезка завершена: попыток {n} из {total} кандидатов",
                                 n=cut_count, total=total))
                self._log_processing_summary(i18n.tr("Нарезка выбранных кандидатов"),
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
            messagebox.showinfo(i18n.tr("Кандидаты"), i18n.tr("Дождитесь завершения нарезки"))
            return
        with self._candidates_lock:
            pending = [c for c in self.candidates if c['status'] == 'pending']
            if not pending:
                messagebox.showinfo(i18n.tr("Кандидаты"), i18n.tr("Нет кандидатов для очистки"))
                return
            if not messagebox.askyesno(i18n.tr("Подтверждение"),
                                       i18n.tr("Удалить {n} кандидатов (ненарезанных)?", n=len(pending))):
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
        self.log(i18n.tr("Кандидаты очищены (удалено {n})", n=len(pending)))

    def save_candidates(self):
        """Сохраняет кандидатов в candidates.yaml (потокобезопасно)"""
        with self._candidates_lock:
            if not self.candidates_file:
                return
            data = {'candidates': [
                {'file': c['file'], 'start': c['start'], 'end': c['end'],
                 'thumbnail': self.get_relative_path(c['thumbnail'])
                 if c.get('thumbnail') else None,
                 'num': c.get('num'),
                 'selected': bool(c.get('selected')),
                 'status': c['status']}
                for c in self.candidates]}
        try:
            os.makedirs(os.path.dirname(self.candidates_file), exist_ok=True)
            with open(self.candidates_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)
        except Exception as e:
            self.log(i18n.tr("Ошибка сохранения кандидатов: {err}", err=e))

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
                        'num': int(c.get('num') or 0),
                        'selected': bool(c.get('selected')),
                        'status': c.get('status', 'pending'),
                    })
            except Exception as e:
                self.log(i18n.tr("Ошибка загрузки кандидатов: {err}", err=e))

    def on_processing_progress(self, file_index, file_count, frame_index, total_frames):
        """Прогресс обработки. Вызывается из рабочего потока, поэтому только считаем
        процент - сам прогрессбар обновляет periodic_update в главном потоке."""
        if file_count <= 0:
            return
        now = time.time()
        # новый файл (или новый запуск) - сбрасываем точку отсчёта ETA
        if getattr(self, '_progress_file_index', None) != file_index or frame_index <= 1:
            self._progress_file_index = file_index
            self._progress_file_t0 = now
            self._progress_file_pos = f"{file_index + 1}/{file_count}"
            if 0 <= file_index < len(self.selected_files):
                self._progress_file_name = os.path.basename(self.selected_files[file_index])
            else:
                self._progress_file_name = ''
        file_progress = frame_index / total_frames if total_frames > 0 else 0.0
        self._progress_percent = 100.0 * (file_index + min(file_progress, 1.0)) / file_count
        self._progress_file_percent = 100.0 * min(file_progress, 1.0)

    @staticmethod
    def _fmt_eta(seconds):
        """Человекочитаемая оценка остатка времени"""
        seconds = max(0, int(seconds))
        if seconds >= 60:
            return i18n.tr("~{m} мин {s} с", m=seconds // 60, s=seconds % 60)
        return i18n.tr("~{s} с", s=seconds)

    def _update_progress_info(self):
        """Строка под прогресс-баром: имя файла + ETA текущего файла"""
        if not self.processing:
            self.progress_info_var.set('')
            return
        parts = []
        name = getattr(self, '_progress_file_name', '')
        pos = getattr(self, '_progress_file_pos', '')
        if name:
            parts.append(f"{name} [{pos}]" if pos else name)
        t0 = getattr(self, '_progress_file_t0', None)
        fp = getattr(self, '_progress_file_percent', 0.0)
        if t0 and fp > 0.5:
            parts.append(i18n.tr("осталось {eta}", eta=self._fmt_eta(
                (time.time() - t0) * (100.0 - fp) / fp)))
        self.progress_info_var.set(' · '.join(parts))

    def setup_logger(self):
        """Настройка логгера"""
        self.logger = logging.getLogger('FreestyleParser')
        self.logger.setLevel(logging.INFO)

        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Хендлер для файла: лог текущей папки дня + стабильный app.log
        # (при смене папки дня лог из папки первой сессии терялся бы)
        log_file = os.path.join(self.output_folder, "processing.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self._day_log_handler = file_handler
        try:
            os.makedirs(self._app_folder, exist_ok=True)
            stable_handler = logging.FileHandler(
                os.path.join(self._app_folder, "app.log"))
            stable_handler.setFormatter(formatter)
            self.logger.addHandler(stable_handler)
        except OSError:
            pass

        # Хендлер для GUI
        self.gui_handler = GuiLogHandler(self)
        self.gui_handler.setFormatter(formatter)
        self.logger.addHandler(self.gui_handler)

    def _redirect_day_log(self):
        """Перенаправляет день-хендлер логгера в текущую папку дня:
        setup_logger вызывается один раз, без этого логи всех дней
        падали в processing.log папки первой сессии."""
        if not self.logger:
            return
        try:
            new_path = os.path.abspath(
                os.path.join(self.output_folder, "processing.log"))
            h = getattr(self, '_day_log_handler', None)
            if h is not None and getattr(h, 'baseFilename', None) == new_path:
                return
            if h is not None:
                self.logger.removeHandler(h)
                h.close()
            fh = logging.FileHandler(new_path)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
            self._day_log_handler = fh
        except OSError:
            pass

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
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = yaml.safe_load(f)
                    if mapping is None:  # Если файл пустой или содержит только комментарии
                        self.athlete_mapping = {}
                    else:
                        # Преобразуем относительные пути в абсолютные
                        self.athlete_mapping = {
                            self._repair_mojibake(name): list(set(attempts))  # Convert back to a list if needed
                            for name, attempts in mapping.items()
                        }
            except Exception as e:
                self.log(i18n.tr("Ошибка при загрузке маппинга: {err}", err=str(e)))
                self.athlete_mapping = {}
        else:
            self.athlete_mapping = {}
        self._prune_athlete_mapping()

    @staticmethod
    def _repair_mojibake(name):
        """Чинит имена, испорченные чтением UTF-8 как cp1251 (Windows-сборки
        до v1.3.1 писали mapping.yaml в utf-8, а читали без encoding - кириллица
        превращалась в 'Ã�Â¿Ã�Â¾Ã�...'). Если обратное преобразование однозначно
        удается - возвращаем чиненое имя, иначе как есть."""
        if not isinstance(name, str) or all(ord(c) < 128 for c in name):
            return name
        for enc in ('cp1251', 'cp1252'):
            try:
                return name.encode(enc).decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return name

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
            self.log(i18n.tr("Убраны мёртвые привязки маппинга: {names}", names=', '.join(pruned)))
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
                self.log(i18n.tr("Ошибка загрузки assignments: {err}", err=e))
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
            self.log(i18n.tr("Ошибка сохранения assignments: {err}", err=e))

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
        self.log(i18n.tr("Подтверждено: {attempt} → {athlete}", attempt=attempt, athlete=athlete))
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
            self.log(i18n.tr("Привязка снята: {attempt}", attempt=attempt))
            self._refresh_attempt_tile(attempt)
            self.update_athlete_list()
            self.refresh_auto_assignments()

    def confirm_all_auto_assignments(self):
        """Подтверждает авто-привязки одним кликом. При активном фильтре -
        только видимые попытки, скрытые остаются провизорными."""
        if not self.attempt_assignments:
            messagebox.showinfo(i18n.tr("Информация"), i18n.tr("Нет авто-привязок для подтверждения."))
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
        self.log(i18n.tr("Подтверждено авто-привязок: {n}", n=n))
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
            self.log(i18n.tr("Ошибка при сохранении маппинга: {err}", err=str(e)))

    def save_ratings(self):
        """Сохраняет рейтинги попыток в файл (может зваться из фонового потока)"""
        ratings_file = os.path.join(self.output_folder, "ratings.yaml")
        with getattr(self, '_ratings_lock', threading.Lock()):
            try:
                with open(ratings_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.attempt_ratings, f, default_flow_style=False, allow_unicode=True)
            except Exception as e:
                self.log(i18n.tr("Ошибка при сохранении рейтингов: {err}", err=str(e)))

    def load_ratings(self):
        """Загружает рейтинги попыток из файла"""
        ratings_file = os.path.join(self.output_folder, "ratings.yaml")
        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r', encoding='utf-8') as f:
                    self.attempt_ratings = yaml.safe_load(f) or {}
            except Exception as e:
                self.log(i18n.tr("Ошибка при загрузке рейтингов: {err}", err=str(e)))
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

            self.log(i18n.tr("Добавлен новый атлет: {name}", name=name))

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
        # заметный шов: иначе на Windows границу панелей не видно и
        # не за что ухватиться мышью (ресайз кажется сломанным)
        self.main_frame.configure(sashwidth=6, sashrelief=tk.RAISED)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === Плавающий выбор языка в правом верхнем углу (в титлбар ОС
        # Tkinter виджеты не помещает) - поверх пустого угла панели «Попытки» ===
        # Компактно: двухбуквенный код (RU/EN) вместо полного названия
        self.header_bar = tk.Frame(self.main_frame)
        self.header_bar.place(relx=1.0, rely=0.0, x=-12, y=4, anchor='ne')
        self.header_lang_label = tk.Label(self.header_bar, text="🌐")
        self.header_lang_label.pack(side=tk.LEFT, padx=(0, 3))
        self.lang_var = tk.StringVar(value=i18n.get_language().upper())
        self._lang_by_code = {code.upper(): code
                              for code, _ in i18n.available_languages()}
        lang_codes = sorted(self._lang_by_code)
        self.lang_menu = ttk.Combobox(
            self.header_bar, textvariable=self.lang_var, values=lang_codes,
            state='readonly', width=4)
        self.lang_menu.pack(side=tk.LEFT)
        self.lang_menu.bind('<<ComboboxSelected>>', self._on_language_selected)

        # === Верхний контейнер с двумя частями ===
        self.top_frame = tk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.main_frame.add(self.top_frame)

        # === ЛЕВЫЙ БЛОК (скроллируемый: на небольших экранах не влезает) ===
        self.left_frame = tk.Frame(self.top_frame, width=460)
        self.top_frame.add(self.left_frame, width=460, minsize=425)

        self.left_canvas = tk.Canvas(self.left_frame, highlightthickness=0)
        left_scroll = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL,
                                    command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_inner = tk.Frame(self.left_canvas)
        self._left_win = self.left_canvas.create_window(
            (0, 0), window=self.left_inner, anchor='nw')
        self.left_inner.bind(
            '<Configure>',
            lambda e: self.left_canvas.configure(
                scrollregion=self.left_canvas.bbox('all')))
        self.left_canvas.bind(
            '<Configure>',
            lambda e: self.left_canvas.itemconfigure(self._left_win, width=e.width))

        # --- Выбор файлов ---
        self.frame_files = tk.LabelFrame(self.left_inner, text=i18n.tr("Файлы"))
        self.frame_files.pack(pady=10, fill=tk.X)

        files_inner = tk.Frame(self.frame_files)
        files_inner.pack(fill=tk.X, padx=5, pady=5)

        # Левая часть — входные файлы (фиксированная ширина)
        files_left = tk.Frame(files_inner)
        files_left.pack(side=tk.LEFT)

        self.label_files = tk.Label(files_left, text=i18n.tr("Входные файлы:"))
        self.label_files.pack(anchor=tk.W)

        self.listbox_files = tk.Listbox(files_left, height=5, width=22)
        self.listbox_files.pack(fill=tk.X, padx=5)
        self.listbox_files.bind('<<ListboxSelect>>', self.on_file_select)

        # подсказка, пока файлы не выбраны (скрывается при выборе)
        self.files_hint_label = tk.Label(
            files_left,
            text=i18n.tr("Выберите файлы с тренировкой\n(поддерживаются .MTS, .MP4, .AVI)"),
            fg='#888888', justify=tk.LEFT, font=('Arial', 8))
        self.files_hint_label.pack(fill=tk.X, padx=5)

        self.button_select_files = tk.Button(files_left, text=i18n.tr("Выбрать файлы"), command=self.select_files)
        self.button_select_files.pack(fill=tk.X, padx=5, pady=5)
        self._attach_tooltip(self.button_select_files,
                             i18n.tr("Выбрать видеофайлы тренировки (.MTS, .MP4, .AVI).\n"
                                     "Можно выбирать прямо с флешки, без копирования."))

        # Правая часть — выходная папка
        files_right = tk.Frame(files_inner)
        files_right.pack(side=tk.LEFT, padx=(10, 0))

        self.label_output = tk.Label(files_right, text=i18n.tr("Выходная папка:"))
        self.label_output.pack(anchor=tk.W)

        # Имя рабочей папки (только чтение) + кнопка открытия в той же строке
        entry_row = tk.Frame(files_right)
        entry_row.pack(fill=tk.X, padx=5, pady=2)

        self.entry_output = tk.Entry(entry_row, width=11, state='readonly', justify='center')
        self.entry_output.pack(side=tk.LEFT)
        self.entry_output.config(state='normal')
        self.entry_output.insert(0, os.path.basename(self.output_folder))
        self.entry_output.config(state='readonly')

        self.button_open_output = tk.Button(entry_row, text=i18n.tr("Открыть"), command=self.open_output_folder)
        self.button_open_output.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(self.button_open_output,
                             i18n.tr("Открыть текущую рабочую папку в проводнике"))

        output_btn_inner = tk.Frame(files_right)
        output_btn_inner.pack(fill=tk.X, padx=5, pady=5)

        self.button_day_folder = tk.Button(output_btn_inner, text="📅",
                                           command=self.choose_day_folder)
        self.button_day_folder.pack(side=tk.LEFT, padx=2)
        self._attach_tooltip(self.button_day_folder,
                             i18n.tr("Выбрать/создать папку дня (дата - часть имени).\n"
                                     "Для каждой тренировки - своя папка."))

        self.button_change_output = tk.Button(output_btn_inner, text=i18n.tr("Изменить"), command=self.change_output_folder)
        self.button_change_output.pack(side=tk.LEFT, padx=2)
        self._attach_tooltip(self.button_change_output,
                             i18n.tr("Выбрать произвольную рабочую папку"))

        # Индикатор состояния рабочей папки
        self.folder_status_label = tk.Label(files_right, text="", anchor=tk.W)
        self.folder_status_label.pack(fill=tk.X, padx=5)

        # --- ROI Canvas ---
        self.frame_roi = tk.LabelFrame(self.left_inner, text=i18n.tr("Область интереса"))
        self.frame_roi.pack(pady=10, fill=tk.X)

        # Кнопки для переключения режимов ROI
        self.frame_roi_controls = tk.Frame(self.frame_roi)
        self.frame_roi_controls.pack(pady=5)
        
        self.button_rectangle_mode = tk.Button(self.frame_roi_controls, text=i18n.tr("Прямоугольник"),
                                              command=lambda: self.switch_roi_mode("rectangle"))
        self.button_rectangle_mode.pack(side=tk.LEFT, padx=5)
        self._attach_tooltip(self.button_rectangle_mode,
                             i18n.tr("Режим прямоугольника: растяните область мышью"))

        self.button_polygon_mode = tk.Button(self.frame_roi_controls, text=i18n.tr("Многоугольник"),
                                            command=lambda: self.switch_roi_mode("polygon"))
        self.button_polygon_mode.pack(side=tk.LEFT, padx=5)
        self._attach_tooltip(self.button_polygon_mode,
                             i18n.tr("Кликайте точки многоугольника, двойной клик - "
                                     "завершить,\nBackspace - убрать последнюю точку"))
        self.button_clear_roi = tk.Button(self.frame_roi_controls, text=i18n.tr("Очистить"),
                                         command=self.clear_roi)
        self.button_clear_roi.pack(side=tk.LEFT, padx=5)
        self._attach_tooltip(self.button_clear_roi, i18n.tr("Стереть область интереса"))

        # подпись режима (для полигона не нужна - подсказка в тултипе кнопки)
        self.label_roi = tk.Label(self.frame_roi, text="")

        self.canvas = tk.Canvas(self.frame_roi, height=CANVAS_HEIGHT, bg="black")
        self.canvas.pack(pady=5, fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)  # Двойной клик завершает многоугольник
        self.canvas.bind("<BackSpace>", self._roi_undo_point)  # Backspace убирает последнюю точку

        # --- Параметры обработки ---
        self.frame_output = tk.LabelFrame(self.left_inner, text=i18n.tr("Настройки обработки"))
        self.frame_output.pack(pady=10, fill=tk.X)

        param_defs = [
            ('min_pause_duration', i18n.tr('Пауза между попытками')),
            ('min_attempt_duration', i18n.tr('Мин. длит. попытки')),
            ('attempt_start_padding', i18n.tr('Запас к началу')),
            ('attempt_end_padding', i18n.tr('Запас к концу')),
            ('min_detection_strength', i18n.tr('Мин. сила детекции (0-1)')),
            ('scan_interval', i18n.tr('Шаг скана, с')),
            ('scan_threshold', i18n.tr('Порог скана (0-1)')),
            ('reid_embed_rate', i18n.tr('OSNet сэмплов/с в попытке')),
            ('precise_left_margin', i18n.tr('Поиск начала влево, с')),
        ]

        # описания + значения по умолчанию для тултипов
        param_hints = {
            'min_pause_duration':
                i18n.tr('Минимальная пауза между попытками, сек реального времени.\n'
                        'Меньше - соседние попытки склеиваются, больше - разрываются.'),
            'min_attempt_duration':
                i18n.tr('Попытки короче отбрасываются, сек.\n'
                        'Больше - отсекаются случайные проходы мимо ROI.'),
            'attempt_start_padding':
                i18n.tr('Сколько секунд записать до начала попытки\n(запас на вход атлета в кадр).'),
            'attempt_end_padding':
                i18n.tr('Сколько секунд записать после попытки\n(запас на выбег).'),
            'min_detection_strength':
                i18n.tr('Минимальная сила сигнала детекции (0-1) для начала/продолжения попытки.\n'
                        'Меньше - ловит тусклые/далёкие появления, больше - строгость.'),
            'scan_interval':
                i18n.tr('Шаг сканирования: 1/2, 1 или 2 сек.\n'
                        'Меньше - точнее границы, больше сэмплов, дольше скан.\n'
                        'Если I-кадры файла реже шага - предупреждение и\n'
                        'рекомендация появятся в логе.'),
            'scan_threshold':
                i18n.tr('Порог YOLO при сканировании (0-1), ниже обычного анализа.\n'
                        'Меньше - больше кандидатов (в т.ч. ложных).'),
            'reid_embed_rate':
                i18n.tr('Сколько OSNet-эмбеддингов в секунду брать внутри попытки\n'
                        'для атрибуции. Больше - точнее, медленнее.'),
            'precise_left_margin':
                i18n.tr('Насколько влево от кандидата искать точное начало попытки, сек.'),
        }

        self.param_vars = {}
        for i, (key, label_text) in enumerate(param_defs):
            col = i % 2
            if col == 0:
                row_frame = tk.Frame(self.frame_output)
                row_frame.pack(fill=tk.X, pady=2, padx=5)
            label = tk.Label(row_frame, text=label_text, anchor=tk.W)
            label.pack(side=tk.LEFT, padx=(0, 2))
            var = tk.StringVar()
            if key == 'scan_interval':
                var.set(_scan_step_to_str(self.processing_params[key]))
                widget = ttk.Combobox(row_frame, textvariable=var,
                                      values=SCAN_STEP_CHOICES, width=4,
                                      state='readonly')
            else:
                var.set(str(self.processing_params[key]))
                widget = tk.Entry(row_frame, textvariable=var, width=6)
            widget.pack(side=tk.LEFT, padx=(0, 15))
            self.param_vars[key] = var
            hint = param_hints.get(key, '')
            if hint:
                self._attach_tooltip(widget, hint + i18n.tr("\n\nПо умолчанию: {default}",
                                     default=DEFAULT_PROCESSING_PARAMS.get(key, '?')))

        # --- Слоумо: чекбокс + коэффициент (в той же сетке параметров) ---
        i = len(param_defs)
        if i % 2 == 0:
            row_frame = tk.Frame(self.frame_output)
            row_frame.pack(fill=tk.X, pady=2, padx=5)
        self.slowmo_var = tk.BooleanVar(value=False)
        self.check_slowmo = tk.Checkbutton(
            row_frame, text=i18n.tr("Слоумо"), variable=self.slowmo_var,
            command=self.on_slowmo_toggle)
        self.check_slowmo.pack(side=tk.LEFT)
        self.slowmo_combo_var = tk.StringVar(value='1/4')
        self.slowmo_combo = ttk.Combobox(
            row_frame, textvariable=self.slowmo_combo_var,
            values=('1/4', '1/8'), width=4, state='disabled')
        self.slowmo_combo.pack(side=tk.LEFT, padx=(2, 0))
        self._attach_tooltip(self.check_slowmo,
                             i18n.tr("Запись в слоумо (например, 96 к/с). Вырезанные\n"
                                     "файлы будут обычной скорости, попытки и\n"
                                     "параметры считаются в реальных секундах."))
        self._attach_tooltip(self.slowmo_combo, i18n.tr("Во сколько раз запись замедлена"))

        # --- Стратегия обработки: 2 чекбокса = 4 комбинации ---
        # [ ] Сразу сохранять  [ ] Определять атлетов
        # ☑☐ - резать всё найденное; ☐☐ - превью-кандидаты для отбора;
        # ☐☑ - сегменты+атлеты в оригиналах (скан атлетов); ☑☑ - резать всё
        # с привязкой к атлетам
        strategy_frame = tk.Frame(self.frame_output)
        strategy_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        tk.Label(strategy_frame, text=i18n.tr("Стратегия:")).pack(side=tk.LEFT, padx=(0, 5))
        self.save_now_var = tk.BooleanVar(value=True)
        self.athletes_var = tk.BooleanVar(value=False)
        cb_save_now = tk.Checkbutton(strategy_frame, text=i18n.tr("Сразу сохранять"),
                                     variable=self.save_now_var,
                                     command=self.on_strategy_change)
        cb_save_now.pack(side=tk.LEFT)
        self._attach_tooltip(cb_save_now,
                             i18n.tr("Вкл: попытки сразу режутся в файлы.\n"
                                     "Выкл: сначала кандидаты, потом «Нарезать»."))
        cb_athletes = tk.Checkbutton(strategy_frame, text=i18n.tr("Определять атлетов"),
                                     variable=self.athletes_var,
                                     command=self.on_strategy_change)
        cb_athletes.pack(side=tk.LEFT, padx=(5, 0))
        self._attach_tooltip(cb_athletes,
                             i18n.tr("Вкл: точный анализ + эмбеддинги + автоатрибуция (медленно).\n"
                                     "Выкл: быстрый скан без атрибуции."))

        # --- Управление процессом ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frame_output, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        # имя текущего файла + ETA (заполняется при обработке)
        self.progress_info_var = tk.StringVar()
        tk.Label(self.frame_output, textvariable=self.progress_info_var,
                 fg='#555', font=('Arial', 8), anchor=tk.W).pack(fill=tk.X)

        control_inner = tk.Frame(self.frame_output)
        control_inner.pack(fill=tk.X)

        self.button_process = tk.Button(control_inner, text=i18n.tr("Обработать"), command=self.start_processing)
        self.button_process.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._attach_tooltip(self.button_process,
                             i18n.tr("Найти и нарезать попытки по выбранным файлам\n"
                                     "(стратегия - чекбоксами выше)"))

        self.button_stop = tk.Button(control_inner, text=i18n.tr("Остановить"), state=tk.DISABLED, command=self.stop_processing)
        self.button_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._attach_tooltip(self.button_stop, i18n.tr("Остановить обработку"))

        # Режим «Просмотр без сохранения»: скан в фоне, попытки живут в памяти,
        # просмотр в плеере, сохранение - по кнопкам (или A/S) в плеере
        self.button_live = tk.Button(control_inner, text=i18n.tr("👁 Просмотр"), command=self.start_live_preview)
        self.button_live.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._attach_tooltip(self.button_live,
                             i18n.tr("Сканировать попытки без сохранения на диск.\n"
                                     "Просмотр в плеере, сохранение - кнопками в плеере\n"
                                     "или клавишами A (предыдущая) / S (текущая).\n"
                                     "«Определять атлетов» вкл -> точный сплиттер (медленно),\n"
                                     "выкл -> грубый быстрый скан (шаг scan_interval)."))


        # === ПРАВЫЙ БЛОК ===
        self.right_frame = tk.PanedWindow(self.top_frame, orient=tk.HORIZONTAL)
        self.top_frame.add(self.right_frame)

        # --- Попытки ---
        self.frame_attempts = tk.LabelFrame(self.right_frame, text=i18n.tr("Попытки"))
        # stretch=always: вся лишняя ширина окна - сетке попыток
        self.right_frame.add(self.frame_attempts, width=950, stretch="always")

        # Область действий
        self.actions_frame = tk.Frame(self.frame_attempts)
        self.actions_frame.pack(fill=tk.X, padx=5, pady=5)

        # Счетчик выбранных попыток
        self.selected_count_label = tk.Label(self.actions_frame, text=i18n.tr("Выбрано: 0"))
        self.selected_count_label.pack(side=tk.LEFT, padx=5)

        # Экспорт выбранных попыток (видео + превью) в стороннюю папку
        self.export_button = tk.Button(self.actions_frame, text=i18n.tr("Экспорт в..."),
                                       state=tk.DISABLED,
                                       command=self.export_selected_attempts)
        self.export_button.pack(side=tk.LEFT, padx=5)
        self._attach_tooltip(self.export_button,
                             i18n.tr("Скопировать выбранные попытки (видео + превью)\n"
                                     "в стороннюю папку"))

        # Фильтры по меткам
        self.rating_filters_frame = tk.Frame(self.actions_frame)
        self.rating_filters_frame.pack(side=tk.LEFT, padx=20)

        # Кнопка фильтра "палец вверх"
        self.filter_up_button = tk.Button(self.rating_filters_frame, text="👍", 
                                        command=lambda: self.toggle_rating_filter("up"),
                                        font=('Arial', 11, 'normal'), bd=2, width=3, height=1)
        self.filter_up_button.pack(side=tk.LEFT, padx=2)
        self._attach_tooltip(self.filter_up_button,
                             i18n.tr("Показывать только попытки с 👍"))

        # Кнопка фильтра "палец вниз"
        self.filter_down_button = tk.Button(self.rating_filters_frame, text="👎",
                                          command=lambda: self.toggle_rating_filter("down"),
                                          font=('Arial', 11, 'normal'), bd=2, width=3, height=1)
        self.filter_down_button.pack(side=tk.LEFT, padx=2)
        self._attach_tooltip(self.filter_down_button,
                             i18n.tr("Показывать только попытки с 👎"))

        # Кнопка "Выбрать все"
        self.select_all_button = tk.Button(self.actions_frame, text=i18n.tr("Выбрать все"), command=self.select_all_attempts)
        self.select_all_button.pack(side=tk.RIGHT, padx=5)
        self._attach_tooltip(self.select_all_button,
                             i18n.tr("Отметить все видимые (по текущему фильтру) попытки"))

        # Кнопка "Снять выделение"
        self.deselect_all_button = tk.Button(self.actions_frame, text=i18n.tr("Снять выделение"), command=self.deselect_all_attempts)
        self.deselect_all_button.pack(side=tk.RIGHT, padx=5)
        self._attach_tooltip(self.deselect_all_button, i18n.tr("Снять все галочки"))

        # Кнопка удаления
        self.delete_button = tk.Button(self.actions_frame, text=i18n.tr("🗑️ Удалить"), command=self.delete_selected_attempts)
        self.delete_button.pack(side=tk.RIGHT, padx=5)
        self._attach_tooltip(self.delete_button,
                             i18n.tr("Удалить выбранные попытки\n(видео и превью-файлы)"))

        # Кнопка нарезки выбранных кандидатов (ручной режим).
        # Ярко выделена, чтобы читалась как кнопка, а не надпись.
        self.button_cut_selected = tk.Button(
            self.actions_frame, text=i18n.tr("✂️ Нарезать"),
            command=self.cut_selected_candidates, state=tk.DISABLED,
            relief=tk.RAISED, bd=2, bg="#dff0d8", activebackground="#c8e6c9",
            font=('Arial', 10, 'bold'), padx=8, pady=2)
        self.button_cut_selected.pack(side=tk.RIGHT, padx=5)
        self._attach_tooltip(self.button_cut_selected,
                             i18n.tr("Нарезать отмеченных кандидатов в видеофайлы"))

        # Создаем canvas и scrollbar для попыток
        self.attempts_canvas = tk.Canvas(self.frame_attempts)
        self.attempts_scrollbar = ttk.Scrollbar(self.frame_attempts, orient="vertical", command=self.attempts_canvas.yview)
        self.attempts_scrollable_frame = tk.Frame(self.attempts_canvas)

        self.attempts_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.attempts_canvas.configure(scrollregion=self.attempts_canvas.bbox("all"))
        )

        self._attempts_win = self.attempts_canvas.create_window(
            (0, 0), window=self.attempts_scrollable_frame, anchor="nw")
        self.attempts_canvas.configure(yscrollcommand=self.attempts_scrollbar.set)
        # динамическое число колонок: пересчёт по ширине канваса
        # (ресайз окна, таскание sash, скрытие/показ атлетов - всё
        # приходит сюда как <Configure>)
        self._attempts_cols_current = 4
        self._attempts_relayout_job = None
        self.attempts_canvas.bind("<Configure>", self._on_attempts_canvas_configure)

        # Размещаем canvas и scrollbar
        self.attempts_canvas.pack(side="left", fill="both", expand=True)
        self.attempts_scrollbar.pack(side="right", fill="y")

        # --- Атлеты ---
        # === Атлеты + Скан атлетов: одна панель, вкладки нарисованы вручную
        # (ttk.Notebook на macOS aqua рисует толстую тематическую рамку
        # вокруг области табов с внутренними отступами - стилем не убираются) ===
        self.frame_athletes = tk.LabelFrame(self.right_frame, text=i18n.tr("Атлеты / Скан"))
        # stretch=never: фиксированная ширина, без пустот по краям
        self.right_frame.add(self.frame_athletes, width=235, stretch="never",
                             minsize=225)
        tabbar = tk.Frame(self.frame_athletes, bd=0, highlightthickness=0)
        tabbar.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._athlete_tab_buttons = {}
        self._athlete_tab_frames = {}
        content = tk.Frame(self.frame_athletes, bd=0, highlightthickness=0)
        content.pack(fill=tk.BOTH, expand=True)
        for key, title in (('athletes', i18n.tr('👤 Атлеты')), ('scan', i18n.tr('🔍 Скан'))):
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

        self.filter_all_button = tk.Button(self.filter_buttons_frame, text=i18n.tr("Все"),
                                           command=lambda: self.set_filter(i18n.tr("Все")))
        self.filter_all_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.filter_all_button, i18n.tr("Показать все попытки"))

        self.filter_unknown_button = tk.Button(self.filter_buttons_frame, text=i18n.tr("Не разобрано"),
                                               command=lambda: self.set_filter(i18n.tr("Неизвестно")))
        self.filter_unknown_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.filter_unknown_button,
                             i18n.tr("Попытки без привязки к атлету"))

        self.filter_confirm_button = tk.Button(self.filter_buttons_frame, text=i18n.tr("На подтверждение"),
                                               command=lambda: self.set_filter(i18n.tr("На подтверждение")))
        self.filter_confirm_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.filter_confirm_button,
                             i18n.tr("Попытки с авто-привязкой, требующие подтверждения"))

        self.confirm_auto_button = tk.Button(self.filter_buttons_frame,
                                             text=i18n.tr("Подтвердить все авто"),
                                             command=self.confirm_all_auto_assignments)
        self.confirm_auto_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.confirm_auto_button,
                             i18n.tr("Подтвердить все авто-привязки разом"))

        self.regenerate_thumbs_button = tk.Button(self.filter_buttons_frame,
                                                  text=i18n.tr("Обновить превью"),
                                                  command=self.start_thumbnail_regeneration)
        self.regenerate_thumbs_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.regenerate_thumbs_button,
                             i18n.tr("Пересоздать скриншоты всех попыток"))

        self.bank_detect_button = tk.Button(self.filter_buttons_frame,
                                            text=i18n.tr("Детекция по банкам"),
                                            command=self.start_bank_detection)
        self.bank_detect_button.pack(fill=tk.X, pady=2)
        self._attach_tooltip(self.bank_detect_button,
                             i18n.tr("Найти атлетов по банкам лиц/жилетов\n(вкладка Скан)"))

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
        self.button_add_athlete = tk.Button(tab_athletes, text=i18n.tr("+ Новый атлет"), command=self.add_athlete)
        self.button_add_athlete.pack(side=tk.BOTTOM, pady=3, padx=2, fill=tk.X)
        self._attach_tooltip(self.button_add_athlete, i18n.tr("Добавить атлета (имя вписывается вручную)"))
        ath_sb.pack(side="right", fill="y")
        self.athletes_canvas.pack(side="left", fill="both", expand=True,
                                  padx=(2, 0), pady=3)

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
        self.update_segments_panel()

        # Инициализируем список атлетов
        self.update_athlete_list()

        # === ЛОГИ ===
        # Логи: сворачиваемая панель, по умолчанию скрыта.
        # Кнопка-язычок живёт ПОВЕРХ основного окна (в скрытой панели
        # она была бы не видна), в левом нижнем углу.
        self.logs_btn = tk.Label(self.main_frame, text=i18n.tr('▸ Логи'),
                                 cursor='hand2', font=('Arial', 10, 'bold'),
                                 padx=4, bg='#e8e8e8')
        self.logs_btn.place(relx=0.0, rely=1.0, x=4, y=-4, anchor='sw')
        self.logs_btn.lift()
        self.logs_btn.bind('<Button-1>', lambda e: self._toggle_logs())
        self.frame_logs_wrap = tk.Frame(self.main_frame)
        self.main_frame.add(self.frame_logs_wrap)
        self.frame_logs = tk.LabelFrame(self.frame_logs_wrap, text=i18n.tr("Логи"))
        self.frame_logs.pack(fill=tk.BOTH, expand=True)
        # панель скрыта по умолчанию (прячем саму pane PanedWindow)
        self._logs_visible = False
        self.main_frame.paneconfigure(self.frame_logs_wrap, hide=True)

        # Копирование из панели логов: на Windows Ctrl+C в readonly Text
        # работает нестабильно - даём явные способы (кнопка + правый клик)
        copy_bar = tk.Frame(self.frame_logs)
        copy_bar.pack(side="top", fill="x")
        def _log_copy_all():
            self.root.clipboard_clear()
            self.root.clipboard_append(self.log_text.get('1.0', 'end-1c'))
            self.log(i18n.tr("Логи скопированы в буфер обмена"))
        copy_btn = tk.Button(copy_bar, text=i18n.tr("📋 Копировать всё"),
                             command=_log_copy_all)
        copy_btn.pack(side="right", padx=4, pady=2)
        self._attach_tooltip(copy_btn,
                             i18n.tr("Скопировать весь лог в буфер обмена\n"
                                     "(правый клик по тексту - копировать выделенное)"))
        def _log_copy_selection():
            try:
                txt = self.log_text.get('sel.first', 'sel.last')
            except tk.TclError:
                return
            if txt:
                self.root.clipboard_clear()
                self.root.clipboard_append(txt)

        # Создаем текстовое поле для логов с прокруткой
        self.log_text = tk.Text(self.frame_logs, height=6, wrap=tk.WORD)
        self.log_scrollbar = ttk.Scrollbar(self.frame_logs, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        # Размещаем текстовое поле и скроллбар
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scrollbar.pack(side="right", fill="y")

        # Делаем текстовое поле только для чтения
        self.log_text.configure(state='disabled')
        log_menu = tk.Menu(self.log_text, tearoff=0)
        log_menu.add_command(label=i18n.tr("Копировать"),
                             command=_log_copy_selection)
        log_menu.add_command(label=i18n.tr("Копировать всё"),
                             command=_log_copy_all)
        self.log_text.bind('<Button-3>',
                           lambda e: log_menu.tk_popup(e.x_root, e.y_root))
        self.log_text.bind('<Control-a>',
                           lambda e: (self.log_text.tag_add('sel', '1.0', 'end-1c'), 'break')[1])

        # Схлопываемые панели левого блока (кнопка '-/+' в шапке)
        for f in (self.frame_files, self.frame_roi, self.frame_output):
            self._make_collapsible(f)

        # Справка '?' в шапке каждого блока - открывает справку
        # на соответствующем разделе docs/help.html
        self._add_help_badge(self.frame_files, 'files',
                             i18n.tr("Справка: Файлы"))
        self._add_help_badge(self.frame_roi, 'roi',
                             i18n.tr("Справка: Область интереса"))
        self._add_help_badge(self.frame_output, 'params',
                             i18n.tr("Справка: Настройки обработки"))
        self._add_help_badge(self.frame_attempts, 'player',
                             i18n.tr("Справка: попытки и плеер"))
        self._add_help_badge(self.frame_athletes, 'athletes',
                             i18n.tr("Справка: Атлеты и Скан"))
        self._add_help_badge(self.frame_logs, 'logs',
                             i18n.tr("Справка: Логи"))

        # Панель 'Атлеты / Скан': скрываемая целиком (кнопка-язычок внизу
        # справа; при скрытии панель попыток расширяется)
        self._ath_pane_visible = True
        self.ath_toggle = tk.Label(self.right_frame, text='⇥', cursor='hand2',
                                   font=('Arial', 9, 'bold'), padx=3,
                                   bg='#e8e8e8')
        self.ath_toggle.place(relx=1.0, rely=1.0, x=-26, y=-4, anchor='se')
        self.ath_toggle.bind('<Button-1>', lambda e: self._toggle_athletes())

        # колесо мыши: один глобальный обработчик для всех скроллируемых панелей
        self._setup_wheel_scrolling()

        # Плавающий выбор языка: поднимаем поверх всего (панели созданы
        # позже него и по z-порядку лежат выше)
        self.header_bar.lift()

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

    def _open_help(self, anchor=None):
        """Открывает HTML-справку в браузере на языке интерфейса
        (docs/help.html - ru, docs/help.en.html - en; в PyInstaller - из
        распакованного бандла). anchor - id раздела (#files, #params, ...).
        macOS отдаёт браузеру file://-URL без query/fragment, поэтому для
        якоря копируем справку во временный файл, вшивая в неё автоскролл
        к разделу; без якоря открываем оригинал как есть."""
        import webbrowser
        base = getattr(sys, '_MEIPASS', None) or os.path.dirname(
            os.path.abspath(__file__))
        # справка на языке интерфейса: help.<lang>.html, fallback - ru
        lang = i18n.get_language()
        help_path = os.path.join(base, 'docs', f'help.{lang}.html')
        if not os.path.exists(help_path):
            help_path = os.path.join(base, 'docs', 'help.html')
        if not os.path.exists(help_path):
            messagebox.showinfo(i18n.tr("Справка"),
                                i18n.tr("Файл справки не найден: {path}", path=help_path))
            return
        open_path = help_path
        if anchor:
            try:
                import tempfile
                with open(help_path, encoding='utf-8') as f:
                    html = f.read()
                inject = (
                    '<script>window.addEventListener("load",function(){'
                    'var e=document.getElementById("' + anchor + '");'
                    'if(e){e.scrollIntoView();}});</script>')
                html = html.replace('</body>', inject + '</body>')
                fd, tmp = tempfile.mkstemp(suffix='.html', prefix='fshelp_')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(html)
                open_path = tmp
            except Exception:
                open_path = help_path   # не смогли - открываем без якоря
        # file://-URL: Windows-путь с backslash и кириллицей требует
        # правильного процентного кодирования
        uri = pathlib.Path(open_path).as_uri()
        if sys.platform == 'darwin':
            # webbrowser/open() на macOS не активирует браузер надёжно;
            # 'open location' - штатный URL-путь; Popen не ждёт -
            # фокус остаётся в браузере, а не возвращается в приложение
            try:
                subprocess.Popen(['osascript', '-e', f'open location "{uri}"'])
                return
            except Exception:
                pass
        try:
            webbrowser.open(uri)
        except Exception:
            webbrowser.open('file:///' + open_path.replace('\\', '/'))

    def _add_help_badge(self, frame, anchor, tooltip=None):
        """Кнопка '?' в шапке блока: открывает справку на разделе anchor
        (позиция - слева от кнопки схлопывания '-/+', как у 'Настроек')"""
        badge = tk.Label(frame, text='?', cursor='hand2',
                         font=('Arial', 10, 'bold'), padx=5, pady=0,
                         bd=1, relief='solid', fg='#0b57d0',
                         bg='#f0f0f0')
        badge.place(relx=1.0, x=-36, y=2, anchor='ne')
        badge.bind('<Button-1>', lambda e: self._open_help(anchor))
        if tooltip:
            self._attach_tooltip(badge, tooltip)
        return badge

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

    def _wheel_target(self, widget):
        """Скроллируемый канвас, в котором находится виджет (по цепочке
        родителей), или None"""
        targets = (self.left_canvas, self.attempts_canvas,
                   self.athletes_canvas, self.segments_canvas)
        w = widget
        while w is not None:
            if w in targets:
                return w
            w = getattr(w, 'master', None)
        return None

    def _setup_wheel_scrolling(self):
        """Один глобальный обработчик колеса: скроллит тот канвас, над которым
        курсор - включая дочерние виджеты панелей (Enter/Leave на канвасе не
        ловил: переход на ребёнок давал Leave и отключал скролл)."""
        debug = bool(os.environ.get('FREESTYLE_WHEEL_DEBUG'))
        dbg_log = os.path.join(self._app_folder, 'wheel_debug.log')

        def dbg(msg):
            if debug:
                try:
                    with open(dbg_log, 'a', encoding='utf-8') as f:
                        f.write(msg + '\n')
                except OSError:
                    pass

        def on_wheel(e):
            try:
                cls = e.widget.winfo_class()
                dbg(f"wheel cls={cls} delta={e.delta!r} w={e.widget} "
                    f"target={self._wheel_target(e.widget)}")
                if cls in ('Listbox', 'Text'):
                    return  # виджет скроллится сам
                canvas = self._wheel_target(e.widget)
                if canvas is not None:
                    canvas.yview_scroll(-1 if getattr(e, 'delta', 0) > 0 else 1,
                                        'units')
            except tk.TclError:
                pass
            except Exception:
                dbg('wheel handler error: ' + traceback.format_exc())

        pend = {'canvas': None, 'dy': 0, 'after_id': None}

        def on_touchpad_scroll(e):
            # Tk 9 на macOS: тачпад/тачпад-жесты мыши дают <TouchpadScroll>,
            # а не <MouseWheel>. Дельта упакована как (dX << 16) | (dY & 0xffff),
            # dY - малые точные значения (±1..±30); направление инвертировано
            # (см. tk::PreciseScrollDeltas в Tk 9).
            # Ивенты идут с частотой тачпада (~120 Гц) - перерисовка канваса
            # на каждый дёргает; накапливаем и применяем раз в ~16 мс.
            try:
                cls = e.widget.winfo_class()
                if cls in ('Listbox', 'Text', 'Scrollbar', 'TScrollbar'):
                    return  # скроллятся сами / нативная обработка
                canvas = self._wheel_target(e.widget)
                if canvas is None:
                    return
                raw = int(getattr(e, 'delta', 0) or 0)
                low = raw & 0xffff
                dy = low if low < 0x8000 else low - 0x10000
                if not dy:
                    return
                pend['canvas'] = canvas
                pend['dy'] += dy
                if pend['after_id'] is None:
                    pend['after_id'] = self.root.after(16, _flush_touchpad)
            except tk.TclError:
                pass
            except Exception:
                dbg('touchpad handler error: ' + traceback.format_exc())

        def _flush_touchpad():
            pend['after_id'] = None
            canvas, dy = pend['canvas'], pend['dy']
            pend['dy'] = 0
            if canvas is None or not dy:
                return
            try:
                # у canvas yview scroll не умеет pixels - скроллим
                # через moveto (как tk::ScrollByPixels в Tk 9)
                frac = canvas.yview()[0]
                h = canvas.winfo_height() or 1
                # скорость прокрутки - половина нативной (по отзыву)
                canvas.yview_moveto(frac - dy / h / 2)
            except tk.TclError:
                pass

        def on_wheel_x11(e):
            try:
                canvas = self._wheel_target(e.widget)
                if canvas is not None:
                    canvas.yview_scroll(-1 if e.num == 4 else 1, 'units')
            except tk.TclError:
                pass
        self.root.bind_all('<MouseWheel>', on_wheel)
        # TouchpadScroll - тип события Tk 9 (macOS CI-сборка собрана на Tk 8.6,
        # там его нет) - bind падал с "bad event type or keysym"
        try:
            self.root.bind_all('<TouchpadScroll>', on_touchpad_scroll)
        except tk.TclError:
            pass
        self.root.bind_all('<Button-4>', on_wheel_x11)
        self.root.bind_all('<Button-5>', on_wheel_x11)

    def _toggle_logs(self):
        """Показать/скрыть панель логов (скрываем pane PanedWindow)"""
        if getattr(self, '_logs_visible', False):
            self.main_frame.paneconfigure(self.frame_logs_wrap, hide=True)
            self.logs_btn.config(text=i18n.tr('▸ Логи'))
            self._logs_visible = False
        else:
            # Tk 9 не перераспределяет pane по reqsize - пересоздаём pane
            # и жёстко фиксируем сайфш: верхняя панель занимает всё, кроме
            # 1/6 высоты окна (панель логов = 1/6 экрана приложения)
            total = self.root.winfo_height()
            h = max(120, int(total / 6))
            self.main_frame.forget(self.frame_logs_wrap)
            self.main_frame.add(self.frame_logs_wrap, height=h, minsize=80,
                                stretch='never')

            def _assert_height():
                try:
                    # Классический PanedWindow при пере-добавлении pane
                    # НЕ растягивает панели - остаётся пустое место.
                    # Явно задаём высоты обеим панелям.
                    self.main_frame.paneconfigure(self.top_frame,
                                                  height=total - h - 6)
                    self.main_frame.paneconfigure(self.frame_logs_wrap,
                                                  height=h)
                except tk.TclError:
                    pass
            self.root.after(30, _assert_height)
            self.root.after(200, _assert_height)
            self.logs_btn.config(text=i18n.tr('▾ Логи'))
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

        self.log(i18n.tr("Установлен фильтр: {filter}", filter=filter_name))

    # --- Динамическая сетка попыток ---

    TILE_W, TILE_H = 200, 150   # размер превью-плитки
    COL_PAD = 5                 # padx плитки в grid
    OVERLAY_FRAC = 0.15         # допустимый оверлей крайней плитки за край

    def _attempts_cols(self):
        """Число колонок сетки по доступной ширине канваса.
        Целые колонки, плюс одна «наполовину влезающая», если её оверлей
        за правый край не превышает OVERLAY_FRAC ширины плитки: небольшой
        срез крайней попытки лучше пустого места, а масштабирование
        превьюшек под ширину колонки заметно дороже по CPU."""
        avail = max(0, self.attempts_canvas.winfo_width() - 20)  # минус скроллбар/запас
        per = self.TILE_W + 2 * self.COL_PAD
        cols = avail // per
        if (cols + 1) * per - avail <= self.OVERLAY_FRAC * self.TILE_W:
            cols += 1
        return max(1, cols)

    def _on_attempts_canvas_configure(self, event):
        # внутренний фрейм - на всю ширину канваса, чтобы сепараторы
        # и сетка не прижимались влево при широком окне
        self.attempts_canvas.itemconfigure(self._attempts_win, width=event.width)
        if self._attempts_relayout_job is not None:
            self.root.after_cancel(self._attempts_relayout_job)
        self._attempts_relayout_job = self.root.after(
            150, self._maybe_relayout_attempts)

    def _maybe_relayout_attempts(self):
        """Перерисовать сетку, только если изменилось число колонок."""
        self._attempts_relayout_job = None
        if not self.root.winfo_exists():
            return
        if self.attempts_canvas.winfo_width() <= 1:
            # окно ещё не отрисовано: стартовое значение (4) не трогаем,
            # реальную ширину даст первый <Configure> при маппинге
            return
        cols = self._attempts_cols()
        if cols != self._attempts_cols_current:
            self._attempts_cols_current = cols
            self.update_attempt_thumbnails()

    def update_attempt_thumbnails(self):
        """Обновляет превью попыток"""
        # сохраняем позицию скролла, чтобы перерисовка её не сбрасывала
        scroll_pos = self.attempts_canvas.yview()
        # Очищаем текущие превью
        for widget in self.attempts_scrollable_frame.winfo_children():
            widget.destroy()
        self._attempt_labels = {}
        self._attempt_attr_frames = {}
        self._rating_buttons = {}

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
        max_cols = self._attempts_cols_current

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
                try:
                    self.generate_thumbnail(attempt, thumbnail_path, self.roi)
                except Exception as e:
                    self.log(i18n.tr("Ошибка превью {attempt}: {err}", attempt=attempt, err=e))

            if os.path.exists(thumbnail_path):
                try:
                    # Загружаем изображение
                    image = Image.open(thumbnail_path)
                    # Изменяем размер
                    image = image.resize((self.TILE_W, self.TILE_H), Image.Resampling.LANCZOS)
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
                    self._rating_buttons[attempt] = (up_button, down_button)
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
                    self.log(i18n.tr("Ошибка при создании превью для {attempt}: {err}", attempt=attempt, err=str(e)))

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
                     text=i18n.tr("—— Кандидаты ({n}): снимите галочки с ненужных, "
                                  "затем «Нарезать» ——", n=len(pending)),
                     fg="#555555").pack(side=tk.LEFT)
            # Очистка ненарезанных кандидатов (для пересканирования с нуля)
            clear_btn = tk.Button(sep_frame, text="🧹", width=3, bd=1,
                                  command=self.clear_pending_candidates)
            clear_btn.pack(side=tk.LEFT, padx=8)
            self._attach_tooltip(clear_btn, i18n.tr("Удалить всех ненарезанных кандидатов\n"
                                                    "(пересканировать с нуля)"))
            crow, ccol = row + 2, 0
            for cand in pending:
                frame = tk.Frame(self.attempts_scrollable_frame)
                frame.grid(row=crow, column=ccol, padx=5, pady=5)

                img_label = None
                thumb = cand.get('thumbnail')
                if thumb and os.path.exists(thumb):
                    try:
                        image = Image.open(thumb)
                        image = image.resize((self.TILE_W, self.TILE_H), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        img_label = tk.Label(frame, image=photo)
                        img_label.image = photo
                        img_label.pack()
                    except Exception as e:
                        self.log(i18n.tr("Ошибка превью кандидата: {err}", err=e))

                name_frame = tk.Frame(frame)
                name_frame.pack()

                var = tk.BooleanVar(value=bool(cand.get('selected')))
                checkbox = tk.Checkbutton(name_frame, variable=var,
                                          command=lambda c=cand, v=var: self.on_candidate_checkbox_change(c, v))
                checkbox.pack(side=tk.LEFT, padx=(0, 5))

                # Компактная подпись: номер кандидата + длительность;
                # исходник и диапазон секунд - в тултипе
                stem = os.path.splitext(os.path.basename(cand['file']))[0]
                num = self._candidate_num(cand)
                est = self._estimate_output_bytes(
                    cand['file'],
                    cand['end'] - cand['start'] + self.processing_params['attempt_start_padding']
                    + self.processing_params['attempt_end_padding']
                    + float(self.processing_params.get('scan_interval', 2.5)) + 1.0)
                size_txt = f" ≈{self._fmt_size(est)}" if est > 0 else ""
                src_missing = not cand.get('file') or \
                    not os.path.exists(self.get_absolute_path(cand['file']))
                if src_missing:
                    size_txt += i18n.tr("  (нет исходника)")
                dur = max(0.0, float(cand['end']) - float(cand['start']))
                name_label = tk.Label(name_frame,
                                      text=i18n.tr("#{num} · {dur:.0f}с",
                                                   num=num, dur=dur) + size_txt,
                                      fg='#d32f2f' if src_missing else None)
                name_label.pack(side=tk.LEFT)
                cand_tip = i18n.tr("{file} · {start:.0f}–{end:.0f}с",
                                   file=stem, start=cand['start'], end=cand['end'])
                self._attach_tooltip(name_label, cand_tip)

                # Двойной клик - видео кандидата из оригинала; правый клик - фотопревью
                for w in (img_label, name_label):
                    if w is not None:
                        w.bind('<Double-Button-1>',
                               lambda e, c=cand: self.show_candidate_video(c))
                        w.bind('<Button-3>',
                               lambda e, c=cand: self.show_candidate_preview(c))
                if img_label is not None:
                    self._attach_tooltip(img_label, cand_tip)

                self.candidate_vars[self._candidate_key(cand)] = var

                ccol += 1
                if ccol >= max_cols:
                    ccol = 0
                    crow += 1

        # --- Live-попытки (просмотр без сохранения) ---
        with self._live_lock:
            live = sorted(self.live_attempts, key=lambda a: a['num'])
        self.live_vars = {}
        if live:
            base_row = (crow + 1) if 'crow' in locals() else (row + 1)
            sep_frame = tk.Frame(self.attempts_scrollable_frame)
            sep_frame.grid(row=base_row, column=0, columnspan=max_cols, pady=(12, 2))
            n_saved = sum(1 for a in live if a['saved'])
            tk.Label(sep_frame,
                     text=i18n.tr("—— Просмотр без сохранения ({n}, "
                                  "сохранено {saved}) ——", n=len(live), saved=n_saved),
                     fg="#555555").pack(side=tk.LEFT)
            save_sel_btn = tk.Button(sep_frame, text=i18n.tr("💾 Сохранить выбранные"),
                                     command=self.save_selected_live)
            save_sel_btn.pack(side=tk.LEFT, padx=8)
            self._attach_tooltip(save_sel_btn,
                                 i18n.tr("Сохранить отмеченные попытки на диск\n"
                                         "(номер = номер в подписи, дырки в нумерации - норма)"))
            clear_live_btn = tk.Button(sep_frame, text="🧹", width=3, bd=1,
                                       command=self.clear_live_attempts)
            clear_live_btn.pack(side=tk.LEFT, padx=4)
            self._attach_tooltip(clear_live_btn,
                                 i18n.tr("Очистить весь список live-попыток\n"
                                         "(на диск ничего не сохранялось, но превью удалятся)"))
            lrow, lcol = base_row + 1, 0
            for i, la in enumerate(live):
                frame = tk.Frame(self.attempts_scrollable_frame)
                frame.grid(row=lrow, column=lcol, padx=5, pady=5)

                img_label = None
                if la['thumb'] and os.path.exists(la['thumb']):
                    try:
                        image = Image.open(la['thumb'])
                        image = image.resize((self.TILE_W, self.TILE_H), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        img_label = tk.Label(frame, image=photo)
                        img_label.image = photo
                        img_label.pack()
                    except Exception as e:
                        self.log(i18n.tr("Ошибка превью live-попытки: {err}", err=e))

                name_frame = tk.Frame(frame)
                name_frame.pack()
                if not la['saved']:
                    # по умолчанию попытка НЕ помечена на сохранение;
                    # галочку ставит S/A в плеере (queued) или пользователь
                    var = tk.BooleanVar(value=bool(la.get('queued')))
                    self.live_vars[la['num']] = var
                    tk.Checkbutton(name_frame, variable=var,
                                   command=lambda n=la['num']: self._on_live_check_toggle(n)
                                   ).pack(side=tk.LEFT, padx=(0, 4))
                status = f"✓ {_live_num_str(la['num'])}.mp4" if la['saved'] \
                    else (i18n.tr("⏳ к сохранению") if la.get('queued') else i18n.tr("— не сохранена"))
                name_label = tk.Label(
                    name_frame,
                    text=i18n.tr("{num} · клип {start:.0f}–{end:.0f}с {status}",
                                 num=_live_num_str(la['num']),
                                 start=la['start'], end=la['end'], status=status),
                    fg='#2e7d32' if la['saved'] else None)
                name_label.pack(side=tk.LEFT)
                for w in (img_label, name_label):
                    if w is not None:
                        w.bind('<Double-Button-1>',
                               lambda e, n=la['num']: self._open_live_by_num(n))
                lcol += 1
                if lcol >= max_cols:
                    lcol = 0
                    lrow += 1

        # подсказка для пустой панели: ни попыток, ни кандидатов, ни live
        if not attempts and not pending and not live:
            hint = tk.Label(self.attempts_scrollable_frame,
                            text=i18n.tr("Здесь появятся попытки после обработки\n"
                                         "Выберите файлы, нарисуйте область интереса "
                                         "и нажмите «Обработать»"),
                            fg='#888888', justify=tk.CENTER,
                            font=('Arial', 10))
            hint.grid(row=0, column=0, columnspan=max_cols, pady=30)

    def save_selected_live(self):
        """Сохранение отмеченных live-попыток (воркер: remux блокирует)"""
        todo = [la for la in list(self.live_attempts)
                if not la['saved'] and la['num'] in getattr(self, 'live_vars', {})
                and self.live_vars[la['num']].get()]
        if not todo:
            self.log(i18n.tr("Нет отмеченных несохранённых live-попыток"))
            return
        self.log(i18n.tr("Сохраняю {n} live-попыток: ",
                         n=len(todo))
                 + ", ".join(_live_num_str(a['num']) for a in todo))

        def worker():
            ok = 0
            for la in todo:
                if self.save_live_attempt(la):
                    ok += 1
            self.log(i18n.tr("Сохранено {ok} из {n} live-попыток", ok=ok, n=len(todo)))
            self.need_update_attempts = True
            self.root.after(0, self.update_attempt_thumbnails)
        threading.Thread(target=worker, daemon=True).start()

    def _on_live_check_toggle(self, num):
        """Синхронизация queued-флага с чекбоксом (клик по галочке в панели)."""
        var = self.live_vars.get(num)
        for la in self.live_attempts:
            if la['num'] == num and not la['saved']:
                la['queued'] = bool(var and var.get())
                break
        self.need_update_attempts = True

    def _flush_live_saves_if_needed(self):
        """Пакетное сохранение отмеченных live-попыток (вызывается при
        закрытии плеера); если нечего сохранять - молча выходим."""
        todo = [la for la in list(self.live_attempts)
                if not la['saved'] and la['num'] in getattr(self, 'live_vars', {})
                and self.live_vars[la['num']].get()]
        if todo:
            self.save_selected_live()

    def _reset_live_state(self):
        """Тихий сброс live-состояния при смене папки дня (без подтверждений):
        список попыток, чекбоксы, live-плеер. Кандидаты другого дня в панель
        новой папки попадать не должны; сохранённые клипы старой папки не трогаем."""
        with self._live_lock:
            had = bool(self.live_attempts)
            self.live_attempts = []
        if not had:
            return
        self._live_player_opened = False
        self._live_next_num = 1
        self.live_vars = {}
        pop = getattr(self, '_preview_popup', None)
        if pop is not None and getattr(pop, '_live', False) and pop.top.winfo_exists():
            pop._closing = True
            pop.close()
        self.log(i18n.tr("Live-список очищен при смене рабочей папки"))

    def clear_live_attempts(self):
        """Полная очистка live-списка: плитки, превью live_*.jpg, плеер"""
        with self._live_lock:
            n = len(self.live_attempts)
            thumbs = [a['thumb'] for a in self.live_attempts]
        if not n:
            return
        if not messagebox.askyesno(
                i18n.tr("Очистить live-попытки"),
                i18n.tr("Удалить список из {n} попыток?\n"
                        "Сохранённые на диск клипы останутся,\n"
                        "несохранённые - будут потеряны.", n=n)):
            return
        with self._live_lock:
            self.live_attempts = []
        for t in thumbs:
            try:
                if t and os.path.exists(t):
                    os.remove(t)
            except OSError as e:
                self.log(i18n.tr("Не удалось удалить превью {file}: {err}", file=t, err=e), logging.WARNING)
        pop = getattr(self, '_preview_popup', None)
        if pop is not None and getattr(pop, '_live', False) and pop.top.winfo_exists():
            pop.close()
        self._live_player_opened = False
        self.need_update_attempts = True
        self.log(i18n.tr("Live-список очищен ({n} попыток)", n=n))

    def _open_live_by_num(self, num):
        """Двойной клик по live-плитке: открыть плеер на этой попытке"""
        with self._live_lock:
            idx = next((i for i, la in enumerate(self.live_attempts)
                        if la['num'] == num), None)
        if idx is not None:
            self.open_live_attempt(idx)

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
            size = i18n.tr("{n:.1f} ГБ", n=total / 1024 ** 3)
        elif total >= 1024 ** 2:
            size = i18n.tr("{n:.0f} МБ", n=total / 1024 ** 2)
        else:
            size = i18n.tr("{n:.0f} КБ", n=total / 1024)
        self.selected_count_label.config(
            text=i18n.tr("Выбрано: {n} ({size})", n=len(self.selected_attempts), size=size))
        btn = getattr(self, 'export_button', None)
        if btn is not None:
            btn.config(state=tk.NORMAL if self.selected_attempts else tk.DISABLED)

    def export_selected_attempts(self):
        """Кнопка 'Экспорт в...': выбор папки и копирование выбранных
        клипов (+ превью jpg) в воркере с прогрессом в логе."""
        if not self.selected_attempts:
            return
        dst = filedialog.askdirectory(title=i18n.tr("Куда экспортировать попытки"))
        if not dst:
            return
        names = sorted(self.selected_attempts)
        total = sum((os.path.getsize(os.path.join(self.output_folder, n))
                     for n in names
                     if os.path.exists(os.path.join(self.output_folder, n))), 0)
        size = i18n.tr("{n:.1f} ГБ", n=total / 1024 ** 3) if total >= 1024 ** 3 \
            else i18n.tr("{n:.0f} МБ", n=total / 1024 ** 2)
        if not messagebox.askyesno(
                i18n.tr("Экспорт"),
                i18n.tr("Скопировать {n} видео (+превью, {size}) в:\n{dst}?",
                        n=len(names), size=size, dst=dst)):
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
                    self.log(i18n.tr("  Ошибка копирования {file}: {err}", file=name, err=e), logging.ERROR)
                    continue
                done += 1
                self.log(i18n.tr("Экспорт {done}/{n}: {file}", done=done, n=len(names), file=name))
            self.log(i18n.tr("Экспорт завершён: {done} из {n} → {dst}",
                             done=done, n=len(names), dst=dst))
            if done:
                self._reveal_in_finder(dst)
        except Exception as e:
            self.log(i18n.tr("Ошибка экспорта: {err}", err=e), logging.ERROR)
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
            self.log(i18n.tr("Не удалось открыть папку {path}: {err}", path=path, err=e), logging.WARNING)

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
                self.log(i18n.tr("Ошибка при удалении {attempt}: {err}", attempt=attempt, err=str(e)), logging.ERROR)
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
            self.log(i18n.tr("Удален файл: {attempt}", attempt=attempt))
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
            messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Не выбрано ни одной попытки для удаления"))
            return

        # Запрашиваем подтверждение
        count = len(self.selected_attempts)
        result = messagebox.askyesno(i18n.tr("Подтверждение"),
                                   i18n.tr("Вы уверены, что хотите удалить {n} попытку(и)?\n"
                                           "Это действие нельзя отменить.", n=count))

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

            messagebox.showinfo(i18n.tr("Успех"), i18n.tr("Удалено {n} попыток", n=len(names)))

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
        """Переключает рейтинг попытки (кнопки на плитке)"""
        self.rate_attempt(attempt, rating_type)

    def rate_attempt(self, attempt, rating_type):
        """Переключает рейтинг и обновляет кнопки на плитке
        (без полной перерисовки панели - плеер остаётся на месте).
        Запись на диск - в фоновом потоке: дисковый I/O в UI-потоке
        подтормаживал видео в плеере."""
        st = self.attempt_ratings.setdefault(attempt, {'up': False, 'down': False})
        st[rating_type] = not st[rating_type]
        buttons = getattr(self, '_rating_buttons', {}).get(attempt)
        if buttons:
            self.update_rating_button_style(buttons[0], buttons[1], st)
        threading.Thread(target=self.save_ratings, daemon=True).start()

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
                messagebox.showinfo(i18n.tr("Превью"), i18n.tr("Клип не найден: {attempt}", attempt=attempt))
                return
            # нарезанные клипы хранятся в обычной скорости
            VideoPreviewPopup(self, src, 0.0, i18n.tr("Попытка {attempt}", attempt=attempt), factor=1,
                              attempt=attempt)
            return
        src = self.get_absolute_path(attempt.source_video)
        if not os.path.exists(src):
            messagebox.showinfo(i18n.tr("Превью"), i18n.tr("Исходный файл не найден"))
            return
        factor = self._preview_slowmo_factor()
        start_real = max(0.0, float(attempt.start) - 1.0)
        dur_real = max(0.0, float(attempt.end) - start_real)
        end_pad = float(self.processing_params.get('attempt_end_padding', 0.5))
        VideoPreviewPopup(self, src,
                          start_real * factor if factor > 1 else start_real,
                          i18n.tr("Попытка {attempt:04d}", attempt=attempt.number), factor=factor,
                          attempt=attempt,
                          interval=(start_real, float(attempt.end) + end_pad),
                          clip_bytes=self._estimate_output_bytes(src, dur_real))

    def show_candidate_video(self, cand):
        """Двойной клик по кандидату: видео из оригинала с позиции кандидата
        (встроенный плеер, hotkeys, Esc). Фотопревью остаётся на правый клик."""
        src = self.get_absolute_path(cand.get('file', ''))
        if not cand.get('file') or not os.path.exists(src):
            messagebox.showinfo(i18n.tr("Превью"), i18n.tr("Исходный файл не найден"))
            return
        factor = self._preview_slowmo_factor()
        start_real = max(0.0, float(cand.get('start', 0)) - 1.0)
        # пауза в конце попытки (+запас на выбег), как в live-просмотре
        interval = (start_real,
                    float(cand.get('end', 0))
                    + float(self.processing_params.get('attempt_end_padding', 0.5)))
        title = self._candidate_title(cand)
        VideoPreviewPopup(self, src,
                          start_real * factor if factor > 1 else start_real,
                          title,
                          factor=factor, interval=interval,
                          clip_bytes=self._estimate_output_bytes(src, interval[1] - interval[0]),
                          cand=cand)

    def show_candidate_preview(self, cand):
        """Превью кандидата в сохранённом (полном) размере - попапом.
        Закрывается кликом или Esc."""
        thumb = cand.get('thumbnail')
        if not thumb or not os.path.exists(thumb):
            messagebox.showinfo(i18n.tr("Превью"), i18n.tr("Превью кандидата не найдено"))
            return
        try:
            img = Image.open(thumb)
        except Exception as e:
            self.log(i18n.tr("Не удалось открыть превью: {err}", err=e))
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
            # tr() возвращает TrStr (ключ+аргументы): переводим в момент
            # наведения - тултипы следуют за сменой языка в рантайме
            shown = i18n.tr(text.key, **text.args) if isinstance(text, i18n.TrStr) else text
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{widget.winfo_rootx() + 12}+"
                           f"{widget.winfo_rooty() + widget.winfo_height() + 4}")
            tk.Label(tw, text=shown, bg="#ffffe0", relief="solid", borderwidth=1,
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
            if sys.platform == 'win32':
                # explorer требует обратные слэши; '/select,' - с запятой
                subprocess.run(['explorer', '/select,', os.path.normpath(attempt)])
            else:
                subprocess.run(['open', '-R', attempt])

    def log(self, message, level=logging.INFO):
        """Добавляет сообщение в лог"""
        if self.logger:
            self.logger.log(level, message)

    def on_click(self, event):
        self.canvas.focus_set()
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

    def _roi_undo_point(self, event=None):
        """Backspace: убрать последнюю точку рисуемого многоугольника"""
        if self.roi_mode != "polygon" or not self.drawing_polygon \
                or not self.roi_points:
            return "break"
        self.roi_points.pop()
        self.canvas.delete("roi_points")
        self.canvas.delete("roi_polygon")
        for x, y in self.roi_points:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                    fill="red", tags="roi_points")
        for i in range(1, len(self.roi_points)):
            p1, p2 = self.roi_points[i - 1], self.roi_points[i]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                    fill="red", width=2, tags="roi_polygon")
        if not self.roi_points:
            self.drawing_polygon = False
        return "break"

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
            self.log(i18n.tr("Многоугольная ROI создана с {n} точками", n=len(self.roi_points)))

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
        self.label_roi.config(text=i18n.tr("Выберите прямоугольную область интереса:")
                              if mode == "rectangle" else "")

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
                messagebox.showwarning(i18n.tr("Предупреждение"), i18n.tr("Не выбраны файлы для обработки."))
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
            self.log(i18n.tr("Режим слоумо включён: видео 1/{factor} скорости "
                             "(например, 96 к/с как 24 к/с). Попытки и параметры "
                             "считаются в реальных секундах, вырезанные файлы будут "
                             "обычной скорости.", factor=factor))
        else:
            self.log(i18n.tr("Режим слоумо выключен."))

    def select_files(self):
        initial_dir = find_default_video_folder()
        files = filedialog.askopenfilenames(
            title=i18n.tr("Выберите видеофайлы"),
            filetypes=[("Video Files", "*.MTS *.MP4 *.AVI")],
            initialdir=initial_dir)
        if not files:
            return
        # порядок диалога непредсказуем (порядок кликов) - сортируем натурально,
        # иначе скан может начаться не с первого файла
        self._set_selected_files(files)
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
                self.log(i18n.tr("Обнаружено слоумо-видео (замедление 1/{factor}) - включён режим слоумо", factor=factor))
            else:
                self.log(i18n.tr("Слоумо по метаданным не обнаружено. Если это слоумо 96 к/с (GH4) - "
                                 "включите галочку \"Слоумо\" вручную"))

    def change_output_folder(self):
        """Выбор произвольной рабочей папки через стандартный диалог ОС"""
        initial = self.output_folder if os.path.isdir(self.output_folder) \
            else os.path.expanduser("~/Desktop/FreestyleParser")
        folder = filedialog.askdirectory(initialdir=initial)
        if folder:
            self._set_output_folder(folder, i18n.tr("Выбрана произвольная папка: {folder}", folder=folder))

    def _sync_params_from_ui(self):
        """Считывает значения параметров из UI в processing_params"""
        for key, var in self.param_vars.items():
            if key == 'scan_interval':
                self.processing_params[key] = _scan_step_from_str(var.get())
                continue
            try:
                val = float(var.get())
                self.processing_params[key] = val
            except ValueError:
                self.log(i18n.tr("Некорректное значение параметра {key}: {value}", key=key, value=var.get()))
                var.set(str(self.processing_params[key]))

    def _sync_ui_from_params(self):
        """Обновляет UI из processing_params"""
        for key, var in self.param_vars.items():
            if key == 'scan_interval':
                var.set(_scan_step_to_str(self.processing_params[key]))
            else:
                var.set(str(self.processing_params[key]))

    def extract_frame(self, video_path, output_path, seek_s=3.0,
                      scale_w=None, scale_h=None, keyframe_only=True):
        """Единая точка извлечения кадра видео в JPEG (ffmpeg).

        - seek_s перед '-i': ffmpeg встаёт на ближайший кей-фрейм, поэтому
          кадр берётся "прямо из кей-фрейма" без полного декодирования.
        - Полный кадр по умолчанию (масштабирование делает UI); scale_w -
          исключение для canvas ROI-редактора; scale_h - жёсткий размер
          canvas_width:CANVAS_HEIGHT (проценты ROI завязаны на него).
        - keyframe_only: '-skip_frame nokey' - декодируются только кей-фреймы,
          самый быстрый путь; при сбое вызывающий код может повторить без него.
        """
        cmd = [get_ffmpeg_path(), '-ss', str(max(0.0, float(seek_s))),
               '-i', self.get_absolute_path(video_path),
               '-vframes', '1', '-q:v', '2']
        if keyframe_only:
            cmd += ['-skip_frame', 'nokey']
        if scale_w and scale_h:
            cmd += ['-vf', f'scale={int(scale_w)}:{int(scale_h)}']
        elif scale_w:
            cmd += ['-vf', f'scale={int(scale_w)}:-1']
        cmd += ['-y', output_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except subprocess.CalledProcessError:
            if float(seek_s) > 0:
                # seek за пределы видео / нет кей-фреймов - берём первый кадр
                fallback = [c for c in cmd]
                fi = fallback.index('-ss')
                fallback[fi + 1] = '0'
                if '-skip_frame' in fallback:
                    ki = fallback.index('-skip_frame')
                    del fallback[ki:ki + 2]
                subprocess.run(fallback, check=True, capture_output=True,
                               timeout=120)
            else:
                raise

    def generate_thumbnail(self, video_path, output_path, roi=None):
        """Превью попытки: полный кадр (без уменьшения, UI сам масштабирует
        при показе) из ближайшего кей-фрейма. Для прямоугольного ROI кадр
        обрезается по ROI в полном разрешении."""
        temp_path = None
        try:
            if self._is_flat_rectangle(roi):  # Прямоугольник (x1, y1, x2, y2 в %)
                temp_path = output_path + ".temp.jpg"
                self.extract_frame(video_path, temp_path, seek_s=3.0)
                import cv2
                frame = cv2.imread(temp_path)
                if frame is None:
                    raise RuntimeError(i18n.tr("ffmpeg не дал кадр для {file}", file=video_path))
                H, W = frame.shape[:2]
                x1, y1, x2, y2 = roi
                crop_x = int(x1 * W / 100)
                crop_y = int(y1 * H / 100)
                crop_w = int((x2 - x1) * W / 100)
                crop_h = int((y2 - y1) * H / 100)
                if crop_w <= 0 or crop_h <= 0:
                    self.log(i18n.tr("Ошибка вырезания скриншота roi={roi}; "
                                     "wxh={W}x{H} - сохраняю полный кадр", roi=roi, W=W, H=H))
                else:
                    frame = frame[crop_y:crop_y + crop_h,
                                  crop_x:crop_x + crop_w]
                if imwrite_unicode(output_path, frame) is False:
                    raise RuntimeError(i18n.tr("cv2.imwrite не записал {file}", file=output_path))
            else:
                # Многоугольник / без ROI: полный кадр без маски (маска
                # применяется только при обработке для детекции)
                self.extract_frame(video_path, output_path, seek_s=3.0)

            return True
        except subprocess.CalledProcessError as e:
            self.log(i18n.tr("Ошибка создания скриншота: {err}", err=e))
            return False
        except Exception as e:
            self.log(i18n.tr("Ошибка создания скриншота {file}: {err}", file=video_path, err=e))
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    self.log(i18n.tr("Не удалось удалить временный файл {file}: {err}", file=temp_path, err=e))

    def smart_thumbnail(self, video_path, output_path):
        """Превью попытки: кадр, где каякер ближе к центру кадра и крупнее.
        Прогоняет YOLO по сэмплам клипа (~2/с, до 120 сэмплов)."""
        try:
            if self._thumb_model is None:
                from splitter import load_yolo_model
                self._thumb_model = load_yolo_model()
            model = self._thumb_model
            cap = open_capture(self.get_absolute_path(video_path))
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
            imwrite_unicode(output_path, best[1])
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
                self.log(i18n.tr("Превью {i}/{n}: {name}", i=i + 1, n=len(files), name=name))
        except Exception as e:
            self.log(i18n.tr("Ошибка регенерации превью: {err}", err=e), logging.ERROR)
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
            messagebox.showinfo(i18n.tr("Информация"), i18n.tr("Идёт обработка - дождитесь завершения."))
            return
        if not any(files for files in self.athlete_mapping.values()) \
                and not self._base_banks():
            messagebox.showinfo(
                i18n.tr("Информация"),
                i18n.tr("Нет банков для детекции. Либо привяжите несколько попыток "
                        "к атлетам (кнопка ✓ или селектор - банк построится из "
                        "подтверждённых привязок), либо подгрузите базовый банк "
                        "(~/.FreestyleParser/banks/)."))
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
            from splitter import load_yolo_model
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
                self.log(i18n.tr("Попыток без профиля: {n} - короткая схема "
                                 "(2-3 кадра из клипа на попытку)", n=len(missing)))
            # прошлые провизорные привязки пересматриваем вместе с новыми
            with self._assignments_lock:
                if self.attempt_assignments:
                    self.attempt_assignments = {}
                    self._save_assignments()
                    self.log(i18n.tr("Прошлые авто-привязки сняты - пересматриваю по банкам"))
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
                    self.log(i18n.tr("  {name}: атлет в клипе не найден (нужен для банка)", name=name))
                    continue
                store[name] = prof
                self._save_profile_store(store)
                self.log(i18n.tr("  Профиль из клипа (банк): {name}", name=name))
            banks = self._athlete_banks(store)
            if not banks:
                self.log(i18n.tr("Банки пусты: нет подтверждённых атлетов и базовых банков. "
                                 "Привяжите попытки вручную или подтвердите сегмент - "
                                 "банк построится из их профилей"), logging.WARNING)
                return
            self.log(i18n.tr("Банки атлетов: {names}", names=', '.join(banks.keys())))
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
                        self.log(i18n.tr("  {name}: атлет в клипе не найден - пропущена", name=name))
                        continue
                    store[name] = prof
                    self._save_profile_store(store)
                    self.log(i18n.tr("  Профиль из клипа: {name}", name=name))
                with self._assignments_lock:
                    if self._try_auto_assign(store, name, banks):
                        n_auto += 1
            self.log(i18n.tr("Детекция по банкам: предложено {n}", n=n_auto))
        except Exception as e:
            self.log(i18n.tr("Ошибка детекции по банкам: {err} {trace}", err=e,
                             trace=traceback.format_exc()),
                     logging.ERROR)
        finally:
            self.processing = False
            self.need_update_attempts = True
            self.root.after(0, lambda: self._after_worker_ui(False))
            self.root.after(0, self.update_athlete_list)

    def _on_selector_assign(self, attempt, choice):
        """Выбор атлета в селекторе на плитке превью."""
        if choice == i18n.tr('➕ Новый / поиск…'):
            name = self._ask_athlete(self.root)
            if not name:
                return
            self.assign_attempt_to_athlete(attempt, name)
            return
        if choice == i18n.tr('— снять'):
            self.remove_assignment(attempt)
        else:
            self.assign_attempt_to_athlete(attempt, choice)

    def _attribution_tip(self, attempt, assigned_athlete, srt):
        """Текст tooltip: кандидаты атрибуции с процентами (или текущий статус)."""
        if assigned_athlete:
            return i18n.tr("{athlete} — подтверждено", athlete=assigned_athlete)
        if not srt:
            return None
        tip = ', '.join(f'{a} {s * 100:.0f}%' for a, s in srt[:2])
        auto = self.attempt_assignments.get(attempt)
        if auto and len(srt) > 1:
            tip += i18n.tr("\nвторой кандидат: {athlete} {percent:.0f}%",
                           athlete=srt[1][0], percent=srt[1][1] * 100)
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
        chip = tk.Label(parent, text=self._short(current or i18n.tr('— выбрать —'), 14),
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
        win.title(i18n.tr("Атлет: выбрать или создать"))
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
        create_btn = tk.Button(bottom, text=i18n.tr('➕ Создать'), bd=1,
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
            create_btn.config(text=i18n.tr("➕ Создать «{name}»", name=self._short(name, 24))
                              if name else i18n.tr('➕ Создать'))

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
            choices = [i18n.tr('— снять'), i18n.tr('➕ Новый / поиск…')] + (names or self._athlete_names())
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
        if choice == i18n.tr('➕ Новый / поиск…'):
            name = self._ask_athlete(self.root)
            if not name:
                return
            self.confirm_segment(seg, name)
            return
        if choice == i18n.tr('— снять'):
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
            btn.configure(text=i18n.tr("🔍 Скан ({pending})", pending=pending) if pending else i18n.tr("🔍 Скан"))
        n_base = len(self._base_banks())
        n_personal = len(self._registry_banks())
        self.bank_status_var.set(
            i18n.tr("Банк: база {n_base} + ваши {n_personal}", n_base=n_base, n_personal=n_personal)
            if (n_base or n_personal) else
            i18n.tr("Банк пуст: подтвердите атлета -\nдальше программа узнает сама"))

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
                     text=i18n.tr("Сегментов нет.\nЗапустите скан атлетов\n(сохранять - выкл, атлеты - вкл)"),
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
            if filt == i18n.tr('Неизвестно') and athlete:
                continue
            if filt == i18n.tr('На подтверждение') and seg.get('confirmed'):
                continue
            if filt not in (i18n.tr('Все'), i18n.tr('Неизвестно'), i18n.tr('На подтверждение')) \
                    and athlete != filt:
                continue
            row = tk.Frame(self.segments_inner)
            row.pack(fill=tk.X, pady=1)
            seg_choices = [i18n.tr('— снять'), i18n.tr('➕ Новый / поиск…')] + names
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
            menu.add_command(label=i18n.tr("Подтвердить: {athlete}", athlete=auto.get('athlete')),
                             command=lambda a=attempt, b=auto.get('athlete'):
                             self.confirm_assignment(a, b))
        athletes = [ath for ath, _ in self._get_attempt_sims(attempt)]
        athletes += [ath for ath in self.athlete_mapping if ath not in athletes]
        for ath in athletes:
            if ath != assigned_athlete:
                menu.add_command(label=i18n.tr("Привязать: {athlete}", athlete=ath),
                                 command=lambda a=attempt, b=ath:
                                 self.assign_attempt_to_athlete(a, b))
        if assigned_athlete or auto:
            menu.add_separator()
            menu.add_command(label=i18n.tr("Снять привязку"),
                             command=lambda a=attempt: self.remove_assignment(a))
        if menu.index(tk.END) is not None and menu.index(tk.END) > 0:
            menu.tk_popup(event.x_root, event.y_root)

    def display_image(self, cv_image):
        # очистка обязательна: иначе новое изображение ляжет поверх старого
        # вместе с ROI-фигурами прошлой папки (переключение дней)
        self.canvas.delete("all")
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
            messagebox.showerror(i18n.tr("Ошибка"), i18n.tr("Папка не найдена."))
        self.processing_config_file = os.path.join(self.output_folder, "processing.yaml")

    def open_video(self, filename, at_real_seconds=None):
        """Открывает видео в VLC/QuickTime; at_real_seconds - реальная секунда
        исходника (для сегментов скана), пересчитывается в файловое время."""
        try:
            # Преобразуем относительный путь в абсолютный
            abs_path = self.get_absolute_path(filename)
            if not os.path.exists(abs_path):
                self.log(i18n.tr("Файл не найден: {file}", file=abs_path))
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

            self.log(i18n.tr("Открыто видео: {file}", file=filename))
        except Exception as e:
            self.log(i18n.tr("Ошибка при открытии видео: {err}", err=str(e)))
            # Пробуем открыть файл системным способом как запасной вариант
            try:
                os.startfile(abs_path)
            except Exception as e2:
                self.log(i18n.tr("Не удалось открыть файл системным способом: {err}", err=str(e2)))

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
            messagebox.showinfo(i18n.tr("Информация"), i18n.tr("Нет активной обработки."))
            return

        self.processing = False
        self.log(i18n.tr("Обработка прервана пользователем."))

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
                                    raw_points = roi_data['polygon']
                                    if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw_points):
                                        # Убеждаемся, что каждая точка является list, а не tuple
                                        self.roi = [list(p) for p in raw_points]
                                    elif len(raw_points) % 2 == 0 \
                                            and all(isinstance(v, (int, float)) for v in raw_points):
                                        # Восстановление "плоского" полигона [x1,y1,x2,y2,...]:
                                        # такой файл нельзя прочитать (падение на старте),
                                        # склеиваем скаляры обратно в пары точек
                                        self.roi = [[raw_points[i], raw_points[i + 1]]
                                                    for i in range(0, len(raw_points), 2)]
                                        self.log(i18n.tr("ROI в processing.yaml был сохранён в плоском формате - восстановлен"))
                                    else:
                                        self.log(i18n.tr("Некорректный ROI в processing.yaml - проигнорирован"),
                                                 logging.WARNING)
                                        self.roi = None
                                    self.roi_mode = "polygon"
                            # Поддержка старого формата для обратной совместимости
                            elif isinstance(roi_data, list):
                                if len(roi_data) == 4:  # Прямоугольник (две точки: x1,y1,x2,y2)
                                    self.roi = tuple(roi_data)  # Преобразуем обратно в tuple
                                    self.roi_mode = "rectangle"
                                elif len(roi_data) > 4 and all(isinstance(point, list) and len(point) == 2 for point in roi_data):  # Многоугольник (список точек)
                                    self.roi = roi_data
                                    self.roi_mode = "polygon"
                    
                    # Загружаем режим ROI только если roi не задан выше -
                    # иначе может возникнуть рассинхрон (плоский roi + polygon)
                    if 'roi_mode' in config['processing-config'] and self.roi is None:
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
                    # (с сортировкой - в yaml мог остаться непредсказуемый порядок)
                    files = config['processing-config'].get('selected_files') or []
                    self._set_selected_files(files)

    def save_processing_config(self):
        """Сохраняет конфигурацию обработки в файл"""
        self._sync_params_from_ui()

        config = {
            'processing-config': {
                'roi_mode': self.roi_mode,
                'params': dict(self.processing_params),
                'slowmo_quarter': bool(self.slowmo_var.get()) if self.slowmo_var else False,
                'slowmo_factor': self._ui_slowmo_factor() if self.slowmo_var else 1,
                'scan_keyframes_only': True,
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
            self.log(i18n.tr("Ошибка отображения canvas: {err}", err=e))
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
        """Показывает canvas.jpg текущей рабочей папки с сохранённым ROI"""
        canvas_path = os.path.join(self.output_folder, "canvas.jpg")
        if not os.path.exists(canvas_path):
            # кадра из этой папки ещё нет - не показываем канвас прошлой папки
            self.canvas.delete("all")
            return False
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

            # Создаем canvas.jpg (кадр из середины видео, уменьшенный
            # до ширины ROI-редактора - через общий экстрактор кадров)
            try:
                self.extract_frame(video_path, canvas_path,
                                   seek_s=seek_time,
                                   scale_w=self.canvas_width,
                                   scale_h=CANVAS_HEIGHT,
                                   keyframe_only=False)
                # Показываем кадр на канве и сохранённый ROI
                self.show_canvas_with_roi(canvas_path)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка создания canvas.jpg (1): {e}")
                return None
            return True
        except Exception as e:
            self.log(i18n.tr("Ошибка создания canvas.jpg (2): {err}", err=e))
            return False

    def get_filtered_attempts(self):
        """Возвращает отфильтрованный список попыток"""
        if not os.path.exists(self.output_folder):
            return []

        files = [f for f in os.listdir(self.output_folder) if f.endswith(".mp4")]
        files = [os.path.join(self.output_folder, f) for f in files]

        # Применяем фильтр по атлетам (mapping хранит базовые имена файлов,
        # сравниваем по basename; во всех ветках возвращаем полные пути)
        if self.filter_var.get() == i18n.tr("Все"):
            filtered_files = files
        elif self.filter_var.get() == i18n.tr("Неизвестно"):
            assigned = {os.path.basename(a)
                        for atts in self.athlete_mapping.values() for a in atts}
            filtered_files = [f for f in files
                              if os.path.basename(f) not in assigned]
        elif self.filter_var.get() == i18n.tr("На подтверждение"):
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
                self.log(i18n.tr("Ошибка при создании превью для перетаскивания: {err}", err=str(e)))

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
        self._reset_live_state()
        self.processing_config_file = os.path.join(self.output_folder, "processing.yaml")
        self.candidates_file = os.path.join(self.output_folder, "candidates.yaml")
        # Создаем выходную папку, если её нет
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        # Новая папка без processing.yaml - чистый лист: ROI и выбор файлов
        # из предыдущей папки дня не переносим (камера/расстановка могли
        # измениться, исходники относятся к другому дню)
        fresh_folder = not os.path.exists(self.processing_config_file)
        if fresh_folder:
            self.roi = None
            self._set_selected_files([])
            self.current_video_for_roi = None
        self.load_processing_config()
        # Настраиваем логгер
        if not self.logger:
            self.setup_logger()
        # лог дня (processing.log) - в текущую папку дня
        if os.path.isdir(self.output_folder):
            self._redirect_day_log()

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

    @staticmethod
    def _freeze_label_width(lbl, samples):
        """Фиксирует ширину Label по самому длинному из вариантов текста:
        динамический текст иначе меняет ширину виджета и дёргает окно плеера."""
        longest = max(samples, key=len)
        fnt = tkfont.Font(font=lbl.cget('font'))
        avg = max(1.0, fnt.measure(longest) / max(1, len(longest)))
        lbl.configure(width=max(1, round(fnt.measure(longest) / avg)))

    def _measure_stream_fps(self, fps_meta):
        """Фактическая частота кадров потока (кадров на секунду файлового
        времени, по POS_MSEC). Метаданные контейнера могут врать: mp4-клип
        из 50i-источника декларирует 50 к/с, а декодер выдаёт 25 - по ним
        плеер играл вдвое быстрее реального времени."""
        cap = self.cap
        ms_first = None
        ms_last = None
        n = 0
        while n < 60:
            if not cap.grab():
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms_first is None:
                ms_first = ms
            ms_last = ms
            n += 1
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
        except Exception:
            pass
        if n < 5 or ms_first is None or ms_last is None or ms_last <= ms_first:
            return fps_meta
        return (n - 1) / ((ms_last - ms_first) / 1000.0)

    def __init__(self, app, video_path, start_file_sec, title, factor=1,
                 attempt=None, live=False, live_index=0, interval=None,
                 clip_bytes=None, cand=None):
        self.app = app
        self._path = video_path
        self.attempt = attempt
        # оценка размера нарезки при просмотре из исходника (None = сам файл)
        self._clip_bytes = clip_bytes or 0
        # Режим live (просмотр без сохранения): навигация по self.app.live_attempts,
        # кнопки сохранения (или клавиши A/S), границы попытки = интервал просмотра.
        # interval=(t0,t1) реальных секунд - для одиночного просмотра (кандидат)
        self._live = live
        self._live_idx = live_index
        # кандидат (просмотр из исходника): переход «Продолжать»/Shift+стрелки
        # по списку ожидающих кандидатов
        self._cand = cand
        self._cand_key = self.app._candidate_key(cand) if cand is not None else None
        self._interval = interval   # (t0, t1) реальных секунд границы просмотра
        self._inside = True     # находимся ли внутри интервала (для границы)
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
        self.cap = open_capture_seekable(video_path)
        if not self.cap.isOpened():
            messagebox.showinfo(i18n.tr("Превью"), i18n.tr("Не удалось открыть видео"))
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        # фактическая частота потока: метаданные mp4-клипов из 50i-источников
        # заявляют 50 к/с, а декодер выдаёт 25 - иначе плеер играет x2
        fps = self._measure_stream_fps(fps)
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
            self._size_str = (i18n.tr("{n:.1f} ГБ", n=size / 1024 ** 3) if size >= 1024 ** 3
                              else i18n.tr("{n:.0f} МБ", n=size / 1024 ** 2))
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
        self._status_lbl = tk.Label(bar, textvariable=self.status_var, anchor='w')
        self._status_lbl.pack(side=tk.LEFT, padx=6)
        # фиксированная ширина: секунды/скорость меняются каждый кадр,
        # иначе окно плеера дёргается
        self._freeze_label_width(self._status_lbl, [
            i18n.tr("{state} x{speed:g} реал "
                    "{time:.1f}{total}{size}",
                    state="⏸", speed=0.0625, time=9999.9,
                    total="/9999.9с", size=" · 9.9 ГБ · попытка ≈99.9 МБ"),
        ])
        ext_btn = tk.Button(bar, text="⧉", width=2,
                            command=self.open_external)
        ext_btn.pack(side=tk.RIGHT, padx=6, pady=3)
        app._attach_tooltip(ext_btn, i18n.tr("Открыть во внешнем плеере\n"
                                             "(macOS - QuickTime, Windows/Linux - VLC)"))

        # Подсказки хоткеев - отдельной строкой: иначе длинная подсказка
        # в одной панели со статусом делает окно шире видео
        hint_bar = tk.Frame(self.top)
        hint_bar.pack(fill=tk.X)
        tk.Label(hint_bar, text=i18n.tr("Space/клик - пауза   ← → ±2с   ↑ ↓ ±10с   "
                                        "⇧↑ x2 (до x16)   ⇧↓ x0.5   "
                                        "⇧→ след. попытка   ⇧← пред.   Esc - закрыть"),
                 fg='#666666').pack(side=tk.LEFT, padx=6)

        # Панель управления: повтор, автопереход, выбор атлета
        ctrl = tk.Frame(self.top)
        ctrl.pack(fill=tk.X)
        self.loop_var = tk.BooleanVar(value=False)
        self.auto_next_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text=i18n.tr("Повтор"), variable=self.loop_var).pack(side=tk.LEFT, padx=4)
        self._next_key = app.next_attempt_after(self.athlete_key) \
            if self.athlete_key else None
        cb = tk.Checkbutton(ctrl, text=i18n.tr("Продолжать просмотр"),
                            variable=self.auto_next_var)
        if self._live:
            # в live-режиме переход - по списку найденных попыток, всегда доступен
            cb.pack(side=tk.LEFT, padx=4)
            # --- кнопки сохранения + индикация текущей/предыдущей ---
            self._save_busy = False
            # варианты текста кнопок сохранения - для фиксации ширины
            self._save_btn_samples = [
                i18n.tr("💾 {word} —", word=w)
                for w in (i18n.tr("Пред."), i18n.tr("Текущая"))
            ] + [
                i18n.tr("✓ {num} сохранена", num="0000"),
                i18n.tr("⏳ {num} запланирована", num="0000"),
                i18n.tr("💾 {word} {num}", word=i18n.tr("Текущая"), num="0000"),
            ]
            self.save_prev_btn = tk.Label(ctrl, text=i18n.tr('💾 Пред. —'), font=('Arial', 9),
                                          cursor='hand2', padx=6, pady=2,
                                          bg='#c8e6c9')
            self.save_prev_btn.bind('<Button-1>', lambda e: self._save_live('prev'))
            self.save_prev_btn.pack(side=tk.LEFT, padx=(14, 4))
            self.save_cur_btn = tk.Label(ctrl, text=i18n.tr('💾 Текущая —'), font=('Arial', 9),
                                         cursor='hand2', padx=6, pady=2,
                                         bg='#c8e6c9')
            self.save_cur_btn.bind('<Button-1>', lambda e: self._save_live('cur'))
            self.save_cur_btn.pack(side=tk.LEFT, padx=4)
            self._freeze_label_width(self.save_prev_btn, self._save_btn_samples)
            self._freeze_label_width(self.save_cur_btn, self._save_btn_samples)
            self._live_status_var = tk.StringVar()
            self._live_status_lbl = tk.Label(ctrl, textvariable=self._live_status_var,
                                             fg='#1565c0', font=('Arial', 9, 'bold'))
            self._live_status_lbl.pack(side=tk.LEFT, padx=8)
            self._freeze_label_width(self._live_status_lbl, [
                i18n.tr("Текущая: {cur} · Пред.: {prev}",
                        cur="0000", prev="0000"),
            ])
        elif self._next_key or self._cand_key is not None:
            cb.pack(side=tk.LEFT, padx=4)
        else:
            cb.pack(side=tk.LEFT, padx=4)
            cb.config(state=tk.DISABLED)
        self._ctrl = ctrl
        if self.attempt is not None and not self._live:
            # рейтинг попытки прямо из плеера (как 👍/👎 на плитке)
            self.rate_up_btn = tk.Button(ctrl, text='👍', width=3, bd=1,
                                         command=lambda: self._rate('up'))
            self.rate_up_btn.pack(side=tk.LEFT, padx=(10, 1))
            self.rate_down_btn = tk.Button(ctrl, text='👎', width=3, bd=1,
                                           command=lambda: self._rate('down'))
            self.rate_down_btn.pack(side=tk.LEFT, padx=1)
            self._refresh_rating_btns()
        if self.attempt is not None:
            self._rebuild_selector()
            # двухфазное удаление: 1-й клик - краснеет, 2-й - удаляет и закрывает
            self._del_armed = False
            # tk.Label, а не tk.Button: на macOS aqua Button игнорирует bg -
            # красная фаза не была бы видна
            self.del_btn = tk.Label(ctrl, text=i18n.tr('🗑 Удалить'), font=('Arial', 9),
                                    cursor='hand2', padx=6, pady=2,
                                    bg='#dddddd')
            self._del_default_bg = self.del_btn.cget('bg')
            self.del_btn.bind('<Button-1>', lambda e: self._delete_two_phase())
            self.del_btn.pack(side=tk.RIGHT, padx=6)

        self._seek_readable(max(0.0, start_file_sec) * 1000)
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
        if self._live:
            # сохранение с клавиатуры: S (ы) - текущая, A (ф) - предыдущая
            self.top.bind('<Key>', self._hotkey_save)
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
        if self._live:
            # live-режим: позиционируемся на выбранную попытку (интервал,
            # заголовок, кнопки) вместо клиппового seek от start_file_sec
            self._load_live_entry(self._live_idx)
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
        # идемпотентность: гасим зависший after предыдущей цепочки.
        # Иначе быстрое пауза/плей (двойной Space, assign из селектора)
        # запускает вторую цепочку - и скорость удваивается
        if getattr(self, '_after_id', None):
            try:
                self.top.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
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
        # live-режим: граница попытки - пауза или переход к следующей.
        # За пределами интервала (перематывали стрелками) играем свободно,
        # повторный вход не ловим, чтобы не зациклиться на границе
        if self._interval is not None and self._inside \
                and self._real_time() >= self._interval[1]:
            self._inside = False
            self._on_boundary()
            return
        self._after_id = self.top.after(max(1, int(self.frame_delay)), self._tick)

    def _on_boundary(self):
        """Достигли конца интервала просмотра (если интервал задан):
        Повтор - с начала интервала; 'Продолжать' - следующая попытка
        (в live-режиме по списку найденных, у кандидатов - по списку
        ожидающих кандидатов, у клипов - по нарезанным клипам); иначе пауза."""
        if self.loop_var.get():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self._interval[0] * self.factor * 1000)
            self._inside = True
            self._tick()
            return
        if self._live and self.auto_next_var.get() \
                and self._live_idx + 1 < len(self.app.live_attempts):
            self._goto(1)
            return
        if not self._live and self.auto_next_var.get() and self._goto(1):
            return
        self.playing = False
        self._update_status()

    def _goto(self, delta):
        """Shift+стрелки: следующая/предыдущая попытка в том же окне.
        live-режим - по списку найденных попыток, у кандидатов - по ожидающим
        кандидатам, иначе по сохранённым клипам. True - переход состоялся."""
        if self._live:
            idx = self._live_idx + delta
            if not (0 <= idx < len(self.app.live_attempts)):
                return False
            self._load_live_entry(idx)
            self._show_after_load()
            return True
        if self._cand_key is not None:
            cands = [c for c in self.app.candidates if c['status'] == 'pending']
            keys = [self.app._candidate_key(c) for c in cands]
            if self._cand_key not in keys:
                return False
            i = keys.index(self._cand_key) + delta
            if not (0 <= i < len(keys)):
                return False
            self._load_candidate(cands[i])
            self._show_after_load()
            return True
        keys = sorted(self.app.attempt_checkboxes.keys())
        if self.athlete_key not in keys:
            return False
        i = keys.index(self.athlete_key) + delta
        if not (0 <= i < len(keys)):
            return False
        nxt = keys[i]
        src = self.app.get_absolute_path(nxt)
        if not os.path.exists(src):
            return False
        self.attempt = self.athlete_key = nxt
        self._load_video(src, 0.0, i18n.tr("Попытка {attempt}", attempt=nxt))
        self._rebuild_selector()
        self._refresh_rating_btns()
        self._show_after_load()
        return True

    def _show_after_load(self):
        """Кадр после перехода к другой попытке в том же окне"""
        if self.top.winfo_exists():
            if self.playing:
                self._tick()
            else:
                ok, frame = self.cap.read()
                if ok:
                    self._show(frame)

    def _load_candidate(self, cand):
        """Загрузка кандидата в то же окно: интервал границы попытки,
        seek на начало, заголовок, оценка размера нарезки."""
        src = self.app.get_absolute_path(cand.get('file', ''))
        start_real = max(0.0, float(cand.get('start', 0)) - 1.0)
        # пауза в конце попытки (+запас на выбег), как при первом открытии
        interval = (start_real,
                    float(cand.get('end', 0))
                    + float(self.app.processing_params.get('attempt_end_padding', 0.5)))
        self._interval = interval
        self._inside = True
        self._cand = cand
        self._cand_key = self.app._candidate_key(cand)
        self._load_video(src,
                         start_real * self.factor if self.factor > 1 else start_real,
                         self.app._candidate_title(cand),
                         clip_bytes=self.app._estimate_output_bytes(src, interval[1] - interval[0]))

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
        """Конец клипа/файла: Повтор - с начала; Продолжать - следующая попытка
        в том же окне (без пересоздания плеера). В live-режиме - по списку
        найденных попыток."""
        if self.loop_var.get():
            start = self._interval[0] if self._interval is not None else 0.0
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * self.factor * 1000)
            self._inside = True
            self._tick()
            return
        if self._live:
            if self.auto_next_var.get() \
                    and self._live_idx + 1 < len(self.app.live_attempts):
                self._goto(1)
                return
            self.playing = False
            self._update_status()
            return
        if self._cand_key is not None:
            if self.auto_next_var.get() and self._goto(1):
                return
            self.playing = False
            self._update_status()
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
            self._load_video(src, 0.0, i18n.tr("Попытка {attempt}", attempt=nxt))
            self._rebuild_selector()
            self._refresh_rating_btns()
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

    def _rate(self, kind):
        """Рейтинг 👍/👎 из плеера + синхронизация кнопок плеера"""
        if not self.athlete_key:
            return
        self.app.rate_attempt(self.athlete_key, kind)
        self._refresh_rating_btns()

    def _refresh_rating_btns(self):
        """Стиль кнопок рейтинга плеера по текущему состоянию попытки"""
        if getattr(self, 'rate_up_btn', None) is None or not self.top.winfo_exists():
            return
        st = self.app.attempt_ratings.get(self.athlete_key) \
            or {'up': False, 'down': False}
        self.app.update_rating_button_style(
            self.rate_up_btn, self.rate_down_btn, st)

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
            self.del_btn.config(text=i18n.tr('🗑 Точно удалить?'), bg='#d32f2f',
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
            self.del_btn.config(text=i18n.tr('🗑 Удалить'), bg=self._del_default_bg,
                                fg='black')
    def _assign(self, choice):
        """Привязка из плеера (общий селектор) + продолжение просмотра"""
        self.app._on_selector_assign(self.athlete_key, choice)
        self._sel_refresh()
        self.app.update_attempt_thumbnails()
        self.playing = True
        self._tick()

    def _seek_readable(self, start_ms):
        """Seek + проверка, что первый кадр реально декодируется. У конца
        AVCHD (длинные GOP) позиция по timestamp может не декодироваться -
        тогда отступаем назад по 0.5с (до 6с). True - кадр прочитан."""
        step = 500
        for i in range(13):
            ms = max(0.0, start_ms - i * step)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, ms)
            ok, _frame = self.cap.read()
            if ok:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, ms)
                return True
        return False

    def _load_video(self, video_path, start_file_sec=0.0, title=None,
                    clip_bytes=None):
        """Загрузка нового клипа в то же окно плеера: чекбоксы, скорость,
        геометрия и позиция окна сохраняются."""
        self._path = video_path
        self._clip_bytes = clip_bytes or 0
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
        self.cap = open_capture_seekable(video_path)
        if not self.cap.isOpened():
            self.close()
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps = self._measure_stream_fps(fps)
        self.base_rate = fps * self.factor
        self.disp_fps = min(self.base_rate, 25.0)
        self.frame_delay = 1000.0 / max(1.0, self.disp_fps)
        n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_real = n_frames / fps / self.factor if fps else 0.0
        try:
            size = os.path.getsize(video_path)
            self._size_str = (i18n.tr("{n:.1f} ГБ", n=size / 1024 ** 3) if size >= 1024 ** 3
                              else i18n.tr("{n:.0f} МБ", n=size / 1024 ** 2))
        except OSError:
            self._size_str = ""
        start_ms = max(0.0, start_file_sec) * 1000
        if fps and n_frames:
            # seek за концом файла даёт пустой экран - клэмпим в границы
            start_ms = min(start_ms, max(0.0, n_frames / fps * 1000 - 500))
        if not self._seek_readable(start_ms):
            # даже с отступом кадр не декодируется - останавливаем, чтобы
            # тик не ушёл в _on_end с пустым экраном
            self.playing = False
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
        total = i18n.tr("/{t:.1f}с", t=self.total_real) if self.total_real > 0 else ""
        size = f" · {self._size_str}" if getattr(self, '_size_str', '') else ""
        if getattr(self, '_clip_bytes', 0) > 0:
            # просмотр из исходника: показываем и размер самой попытки
            size += i18n.tr(" · попытка ≈{size}", size=self.app._fmt_size(self._clip_bytes))
        self.status_var.set(i18n.tr("{state} x{speed:g} реал "
                                    "{time:.1f}{total}{size}",
                                    state=state, speed=self.speed,
                                    time=self._real_time(), total=total, size=size))

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
        # если перемотали внутрь интервала попытки - снова ловим её границу
        if self._interval is not None:
            self._inside = self._real_time() < self._interval[1]
        ok, frame = self.cap.read()
        if ok:
            self._show(frame)

    # --- live-режим: загрузка попытки, сохранение, индикация ---

    def _load_live_entry(self, idx):
        """Загрузка live-попытки: интервал просмотра (с запасами),
        seek на его начало, заголовок, кнопки сохранения."""
        if not (0 <= idx < len(self.app.live_attempts)):
            return
        la = self.app.live_attempts[idx]
        self._live_idx = idx
        sp = float(self.app.processing_params.get('attempt_start_padding', 2))
        ep = float(self.app.processing_params.get('attempt_end_padding', 0.5))
        t0 = max(0.0, la['start'] - sp)
        t1 = la['end'] + ep
        self._interval = (t0, t1)
        src = la['file']
        if not os.path.exists(src):
            self.app.log(i18n.tr("Исходник не найден: {file}", file=src), logging.ERROR)
            self.playing = False
            self._update_status()
            return
        self._load_video(src, t0 * self.factor,
                         i18n.tr("Попытка {attempt} · клип "
                                 "{start:.0f}–{end:.0f}с — просмотр без сохранения",
                                 attempt=_live_num_str(la['num']),
                                 start=la['start'], end=la['end']),
                         clip_bytes=self.app._estimate_output_bytes(src, t1 - t0))
        self._inside = True
        self._refresh_live_ui()

    def _refresh_live_ui(self):
        """Кнопки сохранения + метка 'Текущая: NNNN · Пред.: NNNN'"""
        if not getattr(self, 'save_cur_btn', None) or not self.top.winfo_exists():
            return
        live = self.app.live_attempts
        idx = self._live_idx
        cur = live[idx] if 0 <= idx < len(live) else None
        prev = live[idx - 1] if 1 <= idx <= len(live) else None

        def _set(btn, la, word):
            if la is None:
                btn.config(text=i18n.tr("💾 {word} —", word=word), bg='#eeeeee', fg='#999999')
            elif la.get('saved'):
                btn.config(text=i18n.tr("✓ {num} сохранена", num=_live_num_str(la['num'])),
                           bg='#eeeeee', fg='#2e7d32')
            elif la.get('queued'):
                btn.config(text=i18n.tr("⏳ {num} запланирована", num=_live_num_str(la['num'])),
                           bg='#ffe082', fg='black')
            else:
                btn.config(text=i18n.tr("💾 {word} {num}", word=word, num=_live_num_str(la['num'])),
                           bg='#c8e6c9', fg='black')
        _set(self.save_prev_btn, prev, i18n.tr("Пред."))
        _set(self.save_cur_btn, cur, i18n.tr("Текущая"))
        cur_s = _live_num_str(cur['num']) if cur else '—'
        prev_s = _live_num_str(prev['num']) if prev else '—'
        self._live_status_var.set(i18n.tr("Текущая: {cur} · Пред.: {prev}", cur=cur_s, prev=prev_s))

    def _save_live(self, which):
        """Клик/хоткей сохранения: 'cur' - текущая попытка, 'prev' - предыдущая.
        Remux из UI-потока зависал плеер, поэтому здесь только помечаем попытку
        (чекбокс в панели включается); запись на диск - пакетом после закрытия
        плеера (_flush_live_saves_if_needed) или кнопкой «Сохранить выбранные»."""
        live = self.app.live_attempts
        idx = self._live_idx if which == 'cur' else self._live_idx - 1
        if not (0 <= idx < len(live)):
            return
        la = live[idx]
        if la.get('saved'):
            return
        la['queued'] = True
        lv = self.app.live_vars
        if la['num'] in lv:
            lv[la['num']].set(True)
        else:
            lv[la['num']] = tk.BooleanVar(value=True)
        self.app.need_update_attempts = True
        self._refresh_live_ui()
        self.app.log(i18n.tr("Попытка {attempt} отмечена к сохранению - "
                             "запишется на диск после закрытия плеера",
                             attempt=_live_num_str(la['num'])))

    def _hotkey_save(self, event):
        """S/Ы - сохранить текущую, A/Ф - предыдущую (учёт RU-раскладки)"""
        ch = (event.char or '').lower()
        if ch in ('s', 'ы'):
            self._save_live('cur')
        elif ch in ('a', 'ф'):
            self._save_live('prev')

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
        # live-режим: сохраняем пакетом всё отмеченное в плеере (S/A ставят
        # только чекбокс - remux в UI-потоке зависал)
        if getattr(self, '_live', False):
            try:
                self.app.root.after(0, self.app._flush_live_saves_if_needed)
            except RuntimeError:
                pass


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

def _acquire_single_instance_lock():
    """Защита от второго экземпляра: медленный старт (распаковка exe,
    загрузка моделей) провоцирует повторный клик по иконке - открывалось
    два окна. Держим localhost-сокет на фиксированном порту; если занят -
    приложение уже запущено."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 54321))
        sock.listen(1)
        return sock
    except OSError:
        return None


if __name__ == "__main__":
    lock_sock = _acquire_single_instance_lock()
    if lock_sock is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Freestyle Parser",
            i18n.tr("Программа уже запущена.\n"
                    "Если окна не видно - подождите пару секунд при первом запуске "
                    "или проверьте панель задач."))
        sys.exit(0)
    try:
        root = tk.Tk()
        app = FreestyleParserApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror(i18n.tr("Ошибка запуска"),
                             i18n.tr("Не удалось запустить приложение:\n{err}", err=str(e)))
