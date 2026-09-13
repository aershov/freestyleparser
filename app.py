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
from splitter import process_video, DEFAULT_PROCESSING_PARAMS, load_yolo_model
import sys
import traceback

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
        self.root.title("Freestyle Parser")
        self.root.geometry("1605x900")

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
        self.strategy_var = None  # 'auto' | 'manual', создаётся в setup_ui
        self.candidates = []  # кандидаты попыток ручного режима: {file,start,end,thumbnail,selected,status}
        self.candidate_vars = {}  # ключ кандидата -> tk.BooleanVar чекбокса
        self.candidates_file = None  # путь candidates.yaml, обновляется в on_output_folder_changed
        self._scan_found = 0

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
        self._active_slowmo_factor = 4 if (self.slowmo_var and self.slowmo_var.get()) else 1
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        self.log(f"Запускаем процессинг файлов с roi={self.roi}")
        # Запускаем обработку в отдельном потоке
        if self.strategy_var and self.strategy_var.get() == 'manual':
            threading.Thread(target=self.process_scan, daemon=True).start()
        else:
            threading.Thread(target=self.process_videos, daemon=True).start()

    def _video_has_audio(self, video_path):
        """Проверяет наличие аудио-дорожки (с кэшем по файлу)"""
        if video_path not in self._audio_stream_cache:
            self._audio_stream_cache[video_path] = has_audio_stream(video_path)
        return self._audio_stream_cache[video_path]

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
            # Слоумо: attempt.start/end в реальных секундах, файл длиннее реального
            # времени в slowmo раз (таймстампы растянуты), поэтому файловое время =
            # реальное * slowmo. Перекодируем с setpts, чтобы получить видео обычной
            # скорости с полным fps съёмки (например, 96 к/с).
            file_start = attempt.start * slowmo
            self.log(f"Слоумо 1/{slowmo}: файл {file_start:.2f}s..{file_start + duration * slowmo:.2f}s, "
                     f"перекодирование в обычную скорость")
            cmd = [get_ffmpeg_path(), "-ss", f"{file_start:.3f}", "-i", attempt.source_video,
                   "-t", f"{duration:.3f}",
                   "-vf", f"setpts=PTS/{slowmo}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            if self._video_has_audio(attempt.source_video):
                cmd += ["-af", f"atempo={slowmo}", "-c:a", "aac"]
            cmd += ["-y", output_file]
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
        try:
            next_attempt_number = get_next_attempt_number(self.output_folder)
            process_video(self.selected_files, self.roi, lambda attempt: self.on_attempt_file_created(attempt),
                          next_attempt_number, params=self.processing_params,
                          progress_callback=self.on_processing_progress,
                          should_continue=lambda: self.processing,
                          slowmo_factor=self._active_slowmo_factor)
            self.log(f"Файлы {self.selected_files} успешно обработаны.")
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
        """Обновляет кнопку нарезки выбранных кандидатов"""
        n = sum(1 for c in self.candidates if c['status'] == 'pending' and c.get('selected'))
        if n > 0:
            self.button_cut_selected.config(state=tk.NORMAL, text=f"✂️ Нарезать выбранное ({n})")
        else:
            self.button_cut_selected.config(state=tk.DISABLED, text="✂️ Нарезать выбранное")

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
        self.cut_running = True
        self.processing = True
        self.button_process.config(state=tk.DISABLED)
        self.button_stop.config(state=tk.NORMAL)
        self._progress_percent = 0.0
        self.progress_var.set(0)
        threading.Thread(target=self._cut_candidates_thread, args=(selected,), daemon=True).start()

    def _cut_candidates_thread(self, selected):
        """Точная стадия ручного режима: для каждого кандидата - полный анализ
        в узком окне вокруг него и вырезка найденных попыток."""
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
            for i, cand in enumerate(selected):
                if not self.processing:
                    break
                if any(f == cand['file'] and max(s, cand['start']) < min(e, cand['end'])
                       for f, s, e in cut_windows):
                    self.log(f"Кандидат {i + 1}/{len(selected)} пересекается с уже нарезанной "
                             f"попыткой - пропущен")
                    continue
                win_start = max(0.0, cand['start'] - start_pad - margin)
                win_end = cand['end'] + end_pad + margin
                next_num = get_next_attempt_number(self.output_folder)
                self.log(f"Точный анализ кандидата {i + 1}/{len(selected)}: "
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
            self.log(f"Нарезка завершена: попыток {cut_count} из {len(selected)} кандидатов")
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
        self.log(f"Кандидаты очищены (удалено {len(pending)})")

    def save_candidates(self):
        """Сохраняет кандидатов в candidates.yaml (потокобезопасно)"""
        with self._candidates_lock:
            if not self.candidates_file:
                return
            data = {'candidates': [
                {'file': c['file'], 'start': c['start'], 'end': c['end'],
                 'thumbnail': c.get('thumbnail'), 'selected': bool(c.get('selected')),
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
        self.frame_output = tk.LabelFrame(self.left_frame, text="Настройки")
        self.frame_output.pack(pady=10, fill=tk.X)

        param_defs = [
            ('min_pause_duration', 'Пауза между попытками'),
            ('min_attempt_duration', 'Мин. длит. попытки'),
            ('attempt_start_padding', 'Запас к началу'),
            ('attempt_end_padding', 'Запас к концу'),
            ('min_detection_strength', 'Мин. сила детекции (0-1)'),
            ('scan_interval', 'Шаг скана, с'),
            ('scan_threshold', 'Порог скана (0-1)'),
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

        # --- Слоумо ---
        self.slowmo_var = tk.BooleanVar(value=False)
        self.check_slowmo = tk.Checkbutton(
            self.frame_output,
            text="Слоумо видео 1/4 (напр. Panasonic GH4 96 к/с)",
            variable=self.slowmo_var,
            command=self.on_slowmo_toggle)
        self.check_slowmo.pack(anchor=tk.W, padx=5, pady=(2, 0))

        # --- Стратегия обработки ---
        strategy_frame = tk.Frame(self.frame_output)
        strategy_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        tk.Label(strategy_frame, text="Стратегия:").pack(side=tk.LEFT, padx=(0, 5))
        self.strategy_var = tk.StringVar(value='auto')
        tk.Radiobutton(strategy_frame, text="Нарезать всё", variable=self.strategy_var,
                       value='auto').pack(side=tk.LEFT)
        tk.Radiobutton(strategy_frame, text="Выбрать вручную", variable=self.strategy_var,
                       value='manual').pack(side=tk.LEFT, padx=(5, 0))

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
        self.right_frame.add(self.frame_attempts, width=950)  # ~75% ширины

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

        # Кнопка нарезки выбранных кандидатов (ручной режим)
        self.button_cut_selected = tk.Button(self.actions_frame, text="✂️ Нарезать выбранное",
                                             command=self.cut_selected_candidates, state=tk.DISABLED)
        self.button_cut_selected.pack(side=tk.RIGHT, padx=5)

        # Очистка ненарезанных кандидатов (для пересканирования с нуля)
        self.button_clear_candidates = tk.Button(self.actions_frame, text="🧹 Кандидаты",
                                                 command=self.clear_pending_candidates)
        self.button_clear_candidates.pack(side=tk.RIGHT, padx=5)

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

        # --- Кандидаты ручного режима (ещё не нарезанные) ---
        pending = [c for c in self.candidates if c['status'] == 'pending']
        self.candidate_vars.clear()
        if pending:
            sep = tk.Label(self.attempts_scrollable_frame,
                           text=f"—— Кандидаты ({len(pending)}): снимите галочки с ненужных, затем «Нарезать выбранное» ——",
                           fg="#555555")
            sep.grid(row=row + 1, column=0, columnspan=max_cols, pady=(12, 2))
            crow, ccol = row + 2, 0
            for cand in pending:
                frame = tk.Frame(self.attempts_scrollable_frame)
                frame.grid(row=crow, column=ccol, padx=5, pady=5)

                thumb = cand.get('thumbnail')
                if thumb and os.path.exists(thumb):
                    try:
                        image = Image.open(thumb)
                        image = image.resize((200, 150), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        label = tk.Label(frame, image=photo)
                        label.image = photo
                        label.pack()
                    except Exception as e:
                        self.log(f"Ошибка превью кандидата: {e}")

                name_frame = tk.Frame(frame)
                name_frame.pack()

                var = tk.BooleanVar(value=bool(cand.get('selected')))
                checkbox = tk.Checkbutton(name_frame, variable=var,
                                          command=lambda c=cand, v=var: self.on_candidate_checkbox_change(c, v))
                checkbox.pack(side=tk.LEFT, padx=(0, 5))

                tk.Label(name_frame, text=f"{os.path.basename(cand['file'])} "
                                          f"{cand['start']:.0f}-{cand['end']:.0f}s").pack(side=tk.LEFT)

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

    def on_slowmo_toggle(self):
        """Обработчик переключения чекбокса слоумо"""
        if self.slowmo_var.get():
            self.log("Режим слоумо включён: видео 1/4 скорости (например, 96 к/с как 24 к/с). "
                     "Попытки и параметры считаются в реальных секундах, вырезанные файлы будут обычной скорости.")
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
        self.listbox_files.delete(0, tk.END)
        for file in files:
            self.listbox_files.insert(tk.END, os.path.basename(file))

        # Устанавливаем первое видео как текущее для ROI
        if self.selected_files:
            self.current_video_for_roi = self.selected_files[0]
            self.current_frame_position = 0.5
            self.create_canvas_from_video(self.current_video_for_roi)

            # Пытаемся автоматически определить слоумо по метаданным
            # (для Panasonic GH4 VFR метаданных нет - тогда вручную чекбоксом)
            factor = detect_slowmo_factor(self.selected_files[0])
            self.slowmo_var.set(factor > 1)
            if factor > 1:
                self.log(f"Обнаружено слоумо-видео (замедление 1/{factor}) - включён режим слоумо")
            else:
                self.log("Слоумо по метаданным не обнаружено. Если это слоумо 96 к/с (GH4) - "
                         "включите галочку \"Слоумо видео 1/4\" вручную")

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
                    open POSIX file "{abs_path}"
                    delay 0.5
                    tell document 1
                        try
                            set sound volume to 0.05
                        end try
                        set current time to 0.7
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

                    # Загружаем флаг слоумо
                    if self.slowmo_var is not None and 'slowmo_quarter' in config['processing-config']:
                        self.slowmo_var.set(bool(config['processing-config']['slowmo_quarter']))

                    # Загружаем стратегию обработки
                    if self.strategy_var is not None and 'strategy' in config['processing-config']:
                        strategy = config['processing-config']['strategy']
                        if strategy in ('auto', 'manual'):
                            self.strategy_var.set(strategy)

    def save_processing_config(self):
        """Сохраняет конфигурацию обработки в файл"""
        self._sync_params_from_ui()

        config = {
            'processing-config': {
                'roi_mode': self.roi_mode,
                'params': dict(self.processing_params),
                'slowmo_quarter': bool(self.slowmo_var.get()) if self.slowmo_var else False,
                'strategy': self.strategy_var.get() if self.strategy_var else 'auto',
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
        
        # Загружаем рейтинги
        self.load_ratings()

        # Обновляем интерфейс
        self.update_athlete_list()
        self.update_attempt_thumbnails()
        self.update_folder_status()
        # Показываем canvas и сохранённый ROI новой рабочей папки
        self.refresh_canvas_from_folder()



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
