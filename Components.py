import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
from splitter import process_video
import yaml


class DraggableAttemptFrame(tk.Frame):
    def __init__(self, parent, video_path, master_app):
        super().__init__(parent, bd=1, relief=tk.RAISED)
        self.video_path = video_path
        self.master_app = master_app
        self.filename = os.path.basename(video_path)
        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.drag_window = None

    def setup_drag_events(self, label):
        """Настраивает события перетаскивания для label"""
        # Для мыши
        label.bind('<ButtonPress-1>', self.start_drag)
        label.bind('<ButtonRelease-1>', self.stop_drag)
        label.bind('<Motion>', self.drag_motion)
        
        # Для трекпада
        label.bind('<B1-Motion>', self.drag_motion)
        # Для трехпальцевого жеста
        label.bind('<Control-ButtonPress-1>', self.start_drag)
        label.bind('<Control-ButtonRelease-1>', self.stop_drag)
        label.bind('<Control-B1-Motion>', self.drag_motion)

    def start_drag(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["item"] = self.filename
        self.master_app.drag_data = self.drag_data  # Сохраняем в master_app
        
        # Создаем окно для перетаскивания
        self.drag_window = tk.Toplevel(self)
        self.drag_window.overrideredirect(True)
        self.drag_window.attributes('-alpha', 0.7)
        
        # Копируем содержимое текущего фрейма
        for child in self.winfo_children():
            if isinstance(child, tk.Label):
                # Создаем новый Label
                child_copy = tk.Label(self.drag_window)
                # Копируем все свойства
                for key in child.keys():
                    child_copy[key] = child[key]
                # Копируем изображение
                if hasattr(child, 'image'):
                    child_copy.image = child.image
                child_copy.pack()
        
        # Позиционируем окно
        x = self.winfo_rootx() + event.x
        y = self.winfo_rooty() + event.y
        self.drag_window.geometry(f"+{x}+{y}")
        
        # Меняем курсор
        self.drag_window.config(cursor="hand2")
        self.config(cursor="hand2")

    def drag_motion(self, event):
        if self.drag_window:
            # Обновляем позицию окна
            x = self.winfo_rootx() + event.x
            y = self.winfo_rooty() + event.y
            self.drag_window.geometry(f"+{x}+{y}")

    def stop_drag(self, event):
        if self.drag_window:
            self.drag_window.destroy()
            self.drag_window = None
        self.config(cursor="")
        self.drag_data["item"] = None
        if hasattr(self.master_app, 'drag_data'):
            self.master_app.drag_data = None



class AthleteWidget(tk.Frame):
    def __init__(self, parent, name, master_app):
        super().__init__(parent)
        self.name = name
        self.master_app = master_app
        self.is_selected = False
        self.drag_started = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Создаем кнопку
        self.button = tk.Button(
            self,
            text=name,
            relief=tk.RAISED,
            bg='#f0f0f0',
            activebackground='#e0e0e0'
        )
        self.button.pack(fill=tk.X, expand=True)
        
        # Привязываем обработчики событий
        self.button.bind('<Button-1>', self.on_button_press)
        self.button.bind('<B1-Motion>', self.on_drag)
        self.button.bind('<ButtonRelease-1>', self.on_button_release)
        self.button.bind('<Button-3>', self.show_context_menu)
        
        # Добавляем подсказку
        self.tooltip = None
        self.button.bind('<Enter>', self.show_tooltip)
        self.button.bind('<Leave>', self.hide_tooltip)
        
        # Добавляем счетчик попыток
        self.update_attempt_count()
    
    def on_button_press(self, event):
        """Обработчик нажатия кнопки мыши"""
        self.drag_started = False
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.button.configure(relief=tk.SUNKEN)
    
    def on_drag(self, event):
        """Обработчик перетаскивания"""
        # Если мышь сдвинулась больше чем на 5 пикселей, считаем это началом перетаскивания
        if not self.drag_started and (abs(event.x - self.drag_start_x) > 5 or abs(event.y - self.drag_start_y) > 5):
            self.drag_started = True
            self.master_app.drag_source = self.name
    
    def on_button_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        self.button.configure(relief=tk.RAISED)
        
        if self.drag_started:
            # Если было перетаскивание
            self.master_app.drag_source = None
            self.master_app.update_attempt_thumbnails()
            self.master_app.update_athlete_list()
        else:
            # Если был просто клик
            print(f"DEBUG: on_click called for {self.name}")  # Отладочное сообщение
            self.master_app.set_filter(self.name)
    
    def set_selected(self, selected):
        """Устанавливает состояние выбора"""
        self.is_selected = selected
        if selected:
            self.button.configure(bg='#e0e0ff', activebackground='#d0d0ff')
            self.button.configure(relief=tk.SUNKEN)
        else:
            self.button.configure(bg='#f0f0f0', activebackground='#e0e0e0')
            self.button.configure(relief=tk.RAISED)
    
    def show_context_menu(self, event):
        """Показывает контекстное меню"""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Удалить", command=self.delete_athlete)
        menu.post(event.x_root, event.y_root)
    
    def delete_athlete(self):
        """Удаляет атлета"""
        if messagebox.askyesno("Подтверждение", f"Удалить атлета {self.name}?"):
            # Удаляем из маппинга
            if self.name in self.master_app.athlete_mapping:
                del self.master_app.athlete_mapping[self.name]
            
            # Сохраняем изменения
            self.master_app.save_athlete_mapping()
            
            # Обновляем отображение
            self.master_app.update_attempt_thumbnails()
            self.master_app.update_athlete_list()
            
            # Удаляем виджет
            self.destroy()
    
    def show_tooltip(self, event):
        """Показывает подсказку с количеством попыток"""
        attempts = self.master_app.athlete_mapping.get(self.name, [])
        self.tooltip = tk.Label(
            self,
            text=f"Попыток: {len(attempts)}",
            bg='#ffffe0',
            relief=tk.SOLID,
            borderwidth=1
        )
        self.tooltip.place(x=0, y=self.winfo_height())
    
    def hide_tooltip(self, event):
        """Скрывает подсказку"""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
    
    def update_attempt_count(self):
        """Обновляет счетчик попыток"""
        attempts = self.master_app.athlete_mapping.get(self.name, [])
        self.button.configure(text=f"{self.name} ({len(attempts)})")