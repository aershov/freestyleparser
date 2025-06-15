import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import subprocess
import datetime
import threading
import yaml
import logging
import shutil
from utils import get_ffprobe_path, get_ffmpeg_path

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

class FreestyleParserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Freestyle Parser")

        # Инициализируем переменные
        current_date = datetime.date.today()
        formatted_date = current_date.strftime("%Y-%m-%d")
        self._app_folder = os.path.expanduser(".FreestyleParser")
        if not os.path.exists(self._app_folder):
            os.makedirs(self._app_folder, exist_ok=True)
        self.output_folder = os.path.expanduser(f"~/Desktop/FreestyleParser/{formatted_date}")
        self.processing_config_file = None
        #TODO почистить это всё (
        self.roi = None
        self.logger = None
        self.selected_files = []
        self.processing = False
        self.need_update_attempts = False
        self.athlete_mapping = {}  # Инициализируем пустой словарь
        self.athlete_widgets = []
        self.filter_var = tk.StringVar(value="Все")


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

    def on_attempt_file_created(self, file):
        self.need_update_attempts = True
        self.log(f"Видео попытки готово: {file}")
        # сообщаем, продолжать или нет (если была отмена кнопкой - то остановимся)
        return self.processing

    def process_videos(self):
        """Обрабатывает видео в отдельном потоке"""
        try:
            process_video(self.selected_files, self.output_folder, self.roi, lambda file: self.on_attempt_file_created(file))
            self.log(f"Файлы {self.selected_files} успешно обработан.")
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

        self.label_roi = tk.Label(self.frame_roi, text="Выберите область интереса:")
        self.label_roi.pack()

        self.canvas = tk.Canvas(self.frame_roi, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

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

        # Получаем все попытки
        attempts = [f for f in os.listdir(self.output_folder) if f.endswith(".mp4")]

        # Фильтруем попытки
        filter_name = self.filter_var.get()
        if filter_name != "Все":
            if filter_name == "Неизвестно":
                # Показываем только непривязанные попытки
                attempts = [a for a in attempts if not any(a in v for v in self.athlete_mapping.values())]
            else:
                # Показываем попытки выбранного атлета
                attempts = [self.get_relative_path(p) for p in self.athlete_mapping.get(filter_name, [])]

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

                    # Добавляем имя файла
                    name_label = tk.Label(frame, text=os.path.basename(attempt))
                    name_label.pack()

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
        self.start_x = event.x
        self.start_y = event.y

    def on_drag(self, event):
        self.canvas.delete("roi_rectangle")
        self.end_x = event.x
        self.end_y = event.y
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="roi_rectangle")

    def on_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.roi = (100 * self.start_x/ CANVAS_WIDTH, 100 * self.start_y /CANVAS_HEIGHT, 100 * self.end_x / CANVAS_WIDTH, 100* self.end_y/ CANVAS_HEIGHT)
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="roi_rectangle")

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
        files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.MTS *.MP4 *.AVI")])
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
        folder = filedialog.askdirectory()
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
                # TODO ну что блядь за способ? надо сначала процессинга брать размер и сохранять (
                img = Image.open(temp_path)
                orig_width, orig_height = img.size

                # Конвертируем ROI из процентов в пиксели оригинального размера
                x1, y1, x2, y2 = roi
                crop_x = int(x1 * orig_width / 100)
                crop_y = int(y1 * orig_height / 100)
                crop_w = int((x2 - x1) * orig_width / 100)
                crop_h = int((y2 - y1) * orig_height / 100)

                # Вырезаем ROI из оригинального кадра
                cmd = [
                    get_ffmpeg_path(),
                    '-i', temp_path,
                    '-vf', f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={THUMB_X}:-1',  # Сначала crop, потом scale
                    '-q:v', '2',
                    '-y', output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)
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
                subprocess.Popen(['open', '-a', 'VLC', abs_path])
            elif sys.platform == "win32":  # Windows
                subprocess.Popen(['vlc', abs_path])
            else:  # Linux
                subprocess.Popen(['vlc', abs_path])

            self.log(f"Открыто видео: {filename}")
        except Exception as e:
            self.log(f"Ошибка при открытии видео: {str(e)}")

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
                if 'processing-config' in config and 'roi' in config['processing-config']:
                    # Преобразуем список обратно в tuple
                    roi_list = config['processing-config']['roi']
                    if isinstance(roi_list, list) and len(roi_list) == 4:
                        self.roi = tuple(roi_list)

    def save_processing_config(self):
        """Сохраняет конфигурацию обработки в файл"""
        config = {
            'processing-config': {
                'roi': list(self.roi) if self.roi else None  # Преобразуем tuple в список
            }
        }
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
                        x1, y1, x2, y2 = self.roi
                        self.canvas.create_rectangle(
                            x1 * CANVAS_WIDTH / 100,
                            y1 * CANVAS_HEIGHT / 100,
                            x2 * CANVAS_WIDTH / 100,
                            y2 * CANVAS_HEIGHT / 100,
                            outline="red",
                            tags="roi_rectangle"
                        )
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

        if self.filter_var.get() == "Все":
            return sorted(files)
        elif self.filter_var.get() == "Неизвестно":
            assigned = set(sum(self.athlete_mapping.values(), []))
            return sorted([f for f in files if f not in assigned])
        else:
            return sorted(self.athlete_mapping.get(self.filter_var.get(), []))

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



    def extract_video_frame(self, video_path, output_path, timestamp=0):
        """Извлекает кадр из видео в указанный момент времени"""
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            raise Exception("FFmpeg не найден в системе или в папке bin")
        
        try:
            subprocess.run([
                ffmpeg_path,
                '-ss', str(timestamp),
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',
                output_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Ошибка при извлечении кадра: {e.stderr.decode()}")

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
