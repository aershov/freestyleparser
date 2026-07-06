import numpy as np
import tkinter as tk
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
from splitter import process_video
import sys
import traceback

THUMB_X = 180
THUMB_Y = 120

logging.basicConfig(level=logging.INFO)

CANVAS_WIDTH = 480
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

class FreestyleParserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Freestyle Parser")

        # Инициализируем переменные
        current_date = datetime.date.today()
        formatted_date = current_date.strftime("%Y-%m-%d")
        self._app_folder = os.path.expanduser("~/.FreestyleParser")
        if not os.path.exists(self._app_folder):
            os.makedirs(self._app_folder, exist_ok=True)
        self.output_folder = os.path.expanduser(f"~/Desktop/FreestyleParser/{formatted_date}")
        self.processing_config_file = None
        #TODO почистить это всё (
        self.roi = None  # Теперь это будет список точек многоугольника в процентах
        self.roi_points = []  # Точки многоугольника в пикселях canvas
        self.roi_mode = "polygon"  # "rectangle" или "polygon"
        self.drawing_polygon = False
        self.logger = None
        self.selected_files = []
        self.processing = False
        self.need_update_attempts = False
        self.athlete_mapping = {}  # Инициализируем пустой словарь
        self.athlete_widgets = []
        self.filter_var = tk.StringVar(value="Все")
        self.selected_attempts = set()  # Множество выбранных попыток
        self.attempt_checkboxes = {}  # Словарь для хранения чекбоксов попыток
        self.attempt_ratings = {}  # Словарь для хранения рейтингов попыток: {attempt: {'up': bool, 'down': bool}}
        self.active_rating_filters = set()  # Множество активных фильтров по рейтингу

        # self.on_output_folder_changed()

        # Создаем интерфейс
        self.setup_ui()
        if os.path.exists(self.output_folder):
            self.on_output_folder_changed()
        # Запускаем периодическое обновление
        self.root.after(1000, self.periodic_update)

    def periodic_update(self):
        """Периодическое обновление интерфейса"""
        if self.need_update_attempts:
            self.update_attempt_thumbnails()
            self.need_update_attempts = False
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

    def start_processing(self):
        """Запускает обработку видео"""

        # на случай, если папка по умолчанию, надо добавить проверку и делать это только если output_folder не exists
        self.on_output_folder_changed()
        if not self.selected_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите файлы для обработки")
            return

        if not self.roi:
            messagebox.showwarning("Предупреждение", "Сначала выберите область интереса")
            return
        self.save_processing_config()  # Сохраняем ROI при изменении
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.log(f"Запускаем процессинг файлов с roi={self.roi}")
        # Запускаем обработку в отдельном потоке
        threading.Thread(target=self.process_videos, daemon=True).start()

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
        self.log(f"Saving attempt {attempt.number} from {attempt.start:.2f}s to {attempt.end:.2f}s (duration: {duration:.2f}s)")
        try:
            result = subprocess.run(
                [get_ffmpeg_path(), "-ss", str(attempt.start), "-i", attempt.source_video,
                 "-t", str(duration), "-c", "copy", output_file],
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
        """Обрабатывает видео в отдельном потоке"""
        try:
            next_attempt_number = get_next_attempt_number(self.output_folder)
            process_video(self.selected_files, self.roi, lambda attempt: self.on_attempt_file_created(attempt), next_attempt_number)
            self.log(f"Файлы {self.selected_files} успешно обработаны.")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.log(f"Ошибка обработки: {e} {error_trace}")
        finally:
            self.processing = False
            self.root.after(0, lambda: self.button_process.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.button_stop.config(state=tk.DISABLED))

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

    def save_athlete_mapping(self):
        """Сохраняет маппинг атлетов в файл"""
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
        """Добавляет нового атлета"""
        name = simpledialog.askstring("Новый атлет", "Введите имя атлета:")
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
        """Привязывает попытку к атлету"""
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
        self.update_attempt_thumbnails()
        self.update_athlete_list()

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
        self.left_frame = tk.Frame(self.top_frame)
        self.top_frame.add(self.left_frame)

        # --- Выбор файлов ---
        self.frame_files = tk.LabelFrame(self.left_frame, text="Файлы")
        self.frame_files.pack(pady=10, fill=tk.X)

        self.label_files = tk.Label(self.frame_files, text="Выбранные файлы:")
        self.label_files.pack(anchor=tk.W)

        self.listbox_files = tk.Listbox(self.frame_files, height=5, width=40)
        self.listbox_files.pack(side=tk.LEFT, padx=5)
        self.listbox_files.bind('<<ListboxSelect>>', self.on_file_select)

        self.button_select_files = tk.Button(self.frame_files, text="Выбрать файлы", command=self.select_files)
        self.button_select_files.pack(pady=5)

        self.button_show_screenshot = tk.Button(self.frame_files, text="Область интереса", command=self.show_next_frame)
        self.button_show_screenshot.pack(pady=5)

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

        self.canvas = tk.Canvas(self.frame_roi, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
        self.canvas.pack(pady=5)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)  # Двойной клик завершает многоугольник

        # --- Выходная папка и управление ---
        self.frame_output = tk.LabelFrame(self.left_frame, text="Настройки")
        self.frame_output.pack(pady=10, fill=tk.X)

        self.label_output = tk.Label(self.frame_output, text="Выходная папка:")
        self.label_output.pack(anchor=tk.W)

        output_inner = tk.Frame(self.frame_output)
        output_inner.pack(fill=tk.X)

        self.entry_output = tk.Entry(output_inner, width=30)
        self.entry_output.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_output.insert(0, self.output_folder)

        self.button_change_output = tk.Button(output_inner, text="Изменить", command=self.change_output_folder)
        self.button_change_output.pack(side=tk.LEFT, padx=2)

        self.button_open_output = tk.Button(output_inner, text="Открыть", command=self.open_output_folder)
        self.button_open_output.pack(side=tk.LEFT, padx=2)

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
        self.right_frame.add(self.frame_attempts, width=800)  # 80% ширины

        # Область действий
        self.actions_frame = tk.Frame(self.frame_attempts)
        self.actions_frame.pack(fill=tk.X, padx=5, pady=5)

        # Счетчик выбранных попыток
        self.selected_count_label = tk.Label(self.actions_frame, text="Выбрано: 0")
        self.selected_count_label.pack(side=tk.LEFT, padx=5)

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
        self.frame_athletes = tk.LabelFrame(self.right_frame, text="Атлеты")
        self.right_frame.add(self.frame_athletes, width=200)  # 20% ширины

        # Кнопки фильтров
        self.filter_buttons_frame = tk.Frame(self.frame_athletes)
        self.filter_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        self.filter_all_button = tk.Button(self.filter_buttons_frame, text="Все",
                                           command=lambda: self.set_filter("Все"))
        self.filter_all_button.pack(fill=tk.X, pady=2)

        self.filter_unknown_button = tk.Button(self.filter_buttons_frame, text="Не разобрано",
                                               command=lambda: self.set_filter("Неизвестно"))
        self.filter_unknown_button.pack(fill=tk.X, pady=2)

        # Контейнер для списка атлетов
        self.athletes_container = tk.Frame(self.frame_athletes)
        self.athletes_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.button_add_athlete = tk.Button(self.frame_athletes, text="+ Новый атлет", command=self.add_athlete)
        self.button_add_athlete.pack(pady=5)

        # Инициализируем список атлетов
        self.update_athlete_list()

        # === ЛОГИ ===
        self.frame_logs = tk.LabelFrame(self.main_frame, text="Логи")
        self.main_frame.add(self.frame_logs)

        # Создаем текстовое поле для логов с прокруткой
        self.log_text = tk.Text(self.frame_logs, height=10, wrap=tk.WORD)
        self.log_scrollbar = ttk.Scrollbar(self.frame_logs, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        # Размещаем текстовое поле и скроллбар
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scrollbar.pack(side="right", fill="y")

        # Делаем текстовое поле только для чтения
        self.log_text.configure(state='disabled')

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
        # Очищаем текущие превью
        for widget in self.attempts_scrollable_frame.winfo_children():
            widget.destroy()

        # Получаем отфильтрованные попытки
        filtered_files = self.get_filtered_attempts()
        attempts = [os.path.basename(f) for f in filtered_files]

        # Очищаем выбранные попытки, которые больше не существуют
        self.selected_attempts = {attempt for attempt in self.selected_attempts if attempt in attempts}

        # Очищаем словарь чекбоксов
        self.attempt_checkboxes.clear()

        # Обновляем счетчик
        self.selected_count_label.config(text=f"Выбрано: {len(self.selected_attempts)}")

        # Сортируем попытки по имени
        attempts.sort()

        # Создаем сетку для превью
        row = 0
        col = 0
        max_cols = 4  # Фиксированное количество колонок

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

    def on_attempt_checkbox_change(self, attempt, var):
        """Обработчик изменения состояния чекбокса попытки"""
        if var.get():
            self.selected_attempts.add(attempt)
        else:
            self.selected_attempts.discard(attempt)
        
        # Обновляем счетчик
        self.selected_count_label.config(text=f"Выбрано: {len(self.selected_attempts)}")

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
            deleted_count = 0
            for attempt in self.selected_attempts:
                try:
                    # Получаем полные пути к файлам
                    video_path = os.path.join(self.output_folder, attempt)
                    thumbnail_path = os.path.join(self.output_folder, f"{os.path.splitext(attempt)[0]}.jpg")
                    
                    # Удаляем видео файл
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        deleted_count += 1
                    
                    # Удаляем превью
                    if os.path.exists(thumbnail_path):
                        os.remove(thumbnail_path)
                    
                    # Удаляем из маппинга атлетов
                    for athlete_name, attempts_list in self.athlete_mapping.items():
                        if attempt in attempts_list:
                            attempts_list.remove(attempt)
                    
                    self.log(f"Удален файл: {attempt}")
                    
                except Exception as e:
                    self.log(f"Ошибка при удалении {attempt}: {str(e)}", logging.ERROR)
            
            # Очищаем выбранные попытки
            self.selected_attempts.clear()
            
            # Обновляем счетчик
            self.selected_count_label.config(text=f"Выбрано: {len(self.selected_attempts)}")
            
            # Обновляем интерфейс
            self.update_attempt_thumbnails()
            self.update_athlete_list()
            
            messagebox.showinfo("Успех", f"Удалено {deleted_count} попыток")

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
        self.selected_count_label.config(text=f"Выбрано: {len(self.selected_attempts)}")

    def deselect_all_attempts(self):
        """Снимает выделение со всех попыток"""
        # Очищаем выбранные попытки
        self.selected_attempts.clear()
        
        # Обновляем чекбоксы
        for checkbox_var in self.attempt_checkboxes.values():
            checkbox_var.set(False)
        
        # Обновляем счетчик
        self.selected_count_label.config(text=f"Выбрано: {len(self.selected_attempts)}")

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
            up_button.config(relief=tk.SUNKEN, bg='SystemButtonFace', fg='black', 
                           text='👍', font=('Arial', 9, 'normal'),
                           bd=2, highlightbackground='blue')
        else:
            up_button.config(relief=tk.RAISED, bg='SystemButtonFace', fg='black',
                           text='👍', font=('Arial', 9, 'normal'),
                           bd=1, highlightbackground='SystemButtonFace')
            
        if rating_state['down']:
            down_button.config(relief=tk.SUNKEN, bg='SystemButtonFace', fg='black',
                             text='👎', font=('Arial', 9, 'normal'),
                             bd=2, highlightbackground='blue')
        else:
            down_button.config(relief=tk.RAISED, bg='SystemButtonFace', fg='black',
                             text='👎', font=('Arial', 9, 'normal'),
                             bd=1, highlightbackground='SystemButtonFace')

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
            self.filter_up_button.config(relief=tk.SUNKEN, bg='SystemButtonFace', fg='black',
                                       text='👍', font=('Arial', 11, 'normal'),
                                       bd=2, highlightbackground='blue')
        else:
            self.filter_up_button.config(relief=tk.RAISED, bg='SystemButtonFace', fg='black',
                                       text='👍', font=('Arial', 11, 'normal'),
                                       bd=2, highlightbackground='SystemButtonFace')
            
        if "down" in self.active_rating_filters:
            self.filter_down_button.config(relief=tk.SUNKEN, bg='SystemButtonFace', fg='black',
                                         text='👎', font=('Arial', 11, 'normal'),
                                         bd=2, highlightbackground='blue')
        else:
            self.filter_down_button.config(relief=tk.RAISED, bg='SystemButtonFace', fg='black',
                                         text='👎', font=('Arial', 11, 'normal'),
                                         bd=2, highlightbackground='SystemButtonFace')

    def on_attempt_double_click(self, event, attempt):
        """Обработчик двойного клика по попытке"""
        # Отменяем таймер перетаскивания
        if hasattr(self, 'drag_timer'):
            self.root.after_cancel(self.drag_timer)
            delattr(self, 'drag_timer')

        # Открываем видео
        self.open_video(attempt)

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
            self.roi = (100 * self.start_x/ CANVAS_WIDTH, 100 * self.start_y /CANVAS_HEIGHT, 100 * self.end_x / CANVAS_WIDTH, 100* self.end_y/ CANVAS_HEIGHT)
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
            self.roi = [[100 * x / CANVAS_WIDTH, 100 * y / CANVAS_HEIGHT] for x, y in self.roi_points]
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

    def select_files(self):
        initial_dir = find_default_video_folder()
        files = filedialog.askopenfilenames(
            title="Выберите видеофайлы",
            filetypes=[("Video Files", "*.MTS *.MP4 *.AVI")],
            initialdir=initial_dir)
        if not files:
            return

        self.selected_files = list(files)
        self.listbox_files.delete(0, tk.END)
        for file in files:
            self.listbox_files.insert(tk.END, os.path.basename(file))

        # Устанавливаем первое видео как текущее для ROI
        if self.selected_files:
            self.current_video_for_roi = self.selected_files[0]
            self.current_frame_position = 0.5
            self.create_canvas_from_video(self.current_video_for_roi)

    def change_output_folder(self):
        """Изменяет выходную папку"""
        folder = filedialog.askdirectory(initialdir=os.path.expanduser(f"~/Desktop/FreestyleParser"))
        if folder:
            self.output_folder = folder
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, folder)

            #
            self.on_output_folder_changed()
            self.log(f"Выходная папка изменена на: {folder}")
        else:
            messagebox.showerror("Ошибка", "Папка не найдена.")


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
                    
                    # Изменяем размер и сохраняем
                    frame = cv2.resize(frame, (THUMB_X, THUMB_Y))
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

    def display_image(self, cv_image):
        bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(bgr_image)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.image = self.photo

    def open_output_folder(self):
        folder = self.entry_output.get()
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

    def open_video(self, filename):
        """Открывает видео в VLC"""
        try:
            # Преобразуем относительный путь в абсолютный
            abs_path = self.get_absolute_path(filename)
            if not os.path.exists(abs_path):
                self.log(f"Файл не найден: {abs_path}")
                return

            if sys.platform == "darwin":  # macOS
                # Скрипт открывает файл и сразу запускает воспроизведение
                applescript = f'''
                tell application "QuickTime Player"
                    activate
                    set current_movie to open POSIX file "{abs_path}"
                    set current time of current_movie to 0.7
                    play current_movie
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
                        if os.path.exists(vlc_path):
                            subprocess.Popen([vlc_path, abs_path])
                            vlc_found = True
                            break
                    except Exception:
                        continue
                
                if not vlc_found:
                    # Если VLC не найден, пробуем открыть файл системным способом
                    os.startfile(abs_path)
            else:  # Linux
                subprocess.Popen(['vlc', abs_path])

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

        # Блокируем кнопки
        self.button_stop.config(state=tk.DISABLED)
        self.button_process.config(state=tk.NORMAL)

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

    def save_processing_config(self):
        """Сохраняет конфигурацию обработки в файл"""
        config = {
            'processing-config': {
                'roi_mode': self.roi_mode
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
        
        with open(self.processing_config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)


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
                '-vf', f'scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}',
                '-y',
                canvas_path
            ]

            # Отображаем canvas.jpg
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                # Показываем кадр на канвасе
                if os.path.exists(canvas_path):
                    image = Image.open(canvas_path)
                    self.display_image(np.array(image))
                    # Если есть сохраненный ROI, показываем его
                    if self.roi:
                        if self.roi_mode == "rectangle":
                            x1, y1, x2, y2 = self.roi
                            self.canvas.create_rectangle(
                                x1 * CANVAS_WIDTH / 100,
                                y1 * CANVAS_HEIGHT / 100,
                                x2 * CANVAS_WIDTH / 100,
                                y2 * CANVAS_HEIGHT / 100,
                                outline="red",
                                tags="roi_rectangle"
                            )
                        elif self.roi_mode == "polygon":
                            # Конвертируем точки из процентов в пиксели
                            pixel_points = [(x * CANVAS_WIDTH / 100, y * CANVAS_HEIGHT / 100) for x, y in self.roi]
                            
                            # Рисуем точки
                            for x, y in pixel_points:
                                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tags="roi_points")
                            
                            # Рисуем линии многоугольника
                            for i in range(len(pixel_points)):
                                p1 = pixel_points[i]
                                p2 = pixel_points[(i + 1) % len(pixel_points)]
                                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], 
                                                      fill="red", width=2, tags="roi_polygon")
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

        # Применяем фильтр по атлетам
        if self.filter_var.get() == "Все":
            filtered_files = files
        elif self.filter_var.get() == "Неизвестно":
            assigned = set(sum(self.athlete_mapping.values(), []))
            filtered_files = [f for f in files if f not in assigned]
        else:
            filtered_files = self.athlete_mapping.get(self.filter_var.get(), [])

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

        # Перезагружаем маппинг из новой папки
        self.load_athlete_mapping()
        
        # Загружаем рейтинги
        self.load_ratings()

        # Обновляем интерфейс
        self.update_athlete_list()
        self.update_attempt_thumbnails()



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
