import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, Scale
import librosa
import sounddevice as sd
import pickle
import os
import time
import threading
import soundfile as sf
from datetime import datetime

from audio_classifier import AudioClassifier, CATBOOST_AVAILABLE

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор речевых команд")
        self.classifier = AudioClassifier()
        self.current_word = ""
        self.recordings = []
        self.noise_reduction_level = 0.7
        self.silence_threshold = -40
        self.use_cross_validation = tk.BooleanVar(value=True)
        self.cv_folds = tk.IntVar(value=5)
        self.recording_timer = None
        self.testing_timer = None
        self.recording_seconds = 0
        self.testing_seconds = 0
        self.is_recording = False
        self.is_testing = False
        
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.screen_width = 1920
        self.screen_height = 1080
        self.window_width = 1600
        self.window_height = 900
        
        x = (self.screen_width - self.window_width) // 2
        y = (self.screen_height - self.window_height) // 2
        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        self.setup_ui()
        self.load_existing_dataset()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame, width=self.window_width * 0.6)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_frame = ttk.Frame(main_frame, width=self.window_width * 0.4)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        word_frame = ttk.LabelFrame(left_frame, text="Запись слова", padding=10)
        word_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(word_frame, text="Слово для записи:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
        self.word_entry = ttk.Entry(word_frame, width=20, font=('Arial', 10))
        self.word_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        record_control_frame = ttk.Frame(word_frame)
        record_control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        self.record_btn = ttk.Button(record_control_frame, text="Записать образец (3 сек)", 
                                   command=self.record_sample, style='Accent.TButton')
        self.record_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.record_timer_label = ttk.Label(record_control_frame, text="00:03", font=('Arial', 12, 'bold'), 
                                          foreground="green")
        self.record_timer_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(word_frame, text="Готов к записи", font=('Arial', 9))
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        self.progress = ttk.Progressbar(word_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        settings_frame = ttk.LabelFrame(left_frame, text="Настройки обработки звука", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Уровень шумоподавления:", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W)
        self.noise_scale = Scale(settings_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                length=300, command=self.update_noise_reduction)
        self.noise_scale.set(self.noise_reduction_level)
        self.noise_scale.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Порог обнаружения речи (dB):", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W)
        self.threshold_scale = Scale(settings_frame, from_=-60, to=-20, resolution=5, orient=tk.HORIZONTAL,
                                    length=300, command=self.update_silence_threshold)
        self.threshold_scale.set(self.silence_threshold)
        self.threshold_scale.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        model_frame = ttk.LabelFrame(left_frame, text="Выбор модели классификации", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar(value="random_forest")
        
        model_radio_frame = ttk.Frame(model_frame)
        model_radio_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(model_radio_frame, text="Random Forest", variable=self.model_var, 
                       value="random_forest", command=self.update_model_type).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(model_radio_frame, text="CatBoost", variable=self.model_var, 
                       value="catboost", command=self.update_model_type).grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(model_radio_frame, text="Logistic Regression", variable=self.model_var, 
                       value="logistic_regression", command=self.update_model_type).grid(row=1, column=0, sticky=tk.W)
        
        if not CATBOOST_AVAILABLE:
            ttk.Label(model_radio_frame, text="CatBoost не установлен", foreground="red", font=('Arial', 8)).grid(row=0, column=2, sticky=tk.W)
            self.model_var.set("random_forest")
        
        cv_frame = ttk.LabelFrame(left_frame, text="Настройки валидации", padding=10)
        cv_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(cv_frame, text="Использовать кросс-валидацию", 
                       variable=self.use_cross_validation).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(cv_frame, text="Количество фолдов:", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W)
        cv_spinbox = ttk.Spinbox(cv_frame, from_=3, to=10, width=5, textvariable=self.cv_folds)
        cv_spinbox.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        control_frame = ttk.LabelFrame(left_frame, text="Управление моделью", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_row1 = ttk.Frame(control_frame)
        control_row1.pack(fill=tk.X, pady=5)
        
        self.train_btn = ttk.Button(control_row1, text="Обучить модель", 
                                  command=self.train_model, style='Accent.TButton')
        self.train_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        test_control_frame = ttk.Frame(control_row1)
        test_control_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.test_btn = ttk.Button(test_control_frame, text="Тестировать запись", 
                                 command=self.test_recording)
        self.test_btn.pack(side=tk.LEFT)
        
        self.test_timer_label = ttk.Label(test_control_frame, text="00:03", font=('Arial', 10, 'bold'), 
                                        foreground="blue")
        self.test_timer_label.pack(side=tk.LEFT, padx=(10, 0))
        
        control_row2 = ttk.Frame(control_frame)
        control_row2.pack(fill=tk.X, pady=5)
        
        self.clear_btn = ttk.Button(control_row2, text="Очистить все данные", 
                                  command=self.clear_data)
        self.clear_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.delete_last_btn = ttk.Button(control_row2, text="Удалить последнюю запись", 
                                       command=self.delete_last_recording)
        self.delete_last_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.load_dataset_btn = ttk.Button(control_row2, text="Загрузить датасет", 
                                        command=self.load_existing_dataset)
        self.load_dataset_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        viz_frame = ttk.LabelFrame(left_frame, text="Визуализация аудиозаписи", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_audio_visualization(viz_frame)
        
        stats_frame = ttk.LabelFrame(right_frame, text="Статистика записей", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=12, width=40, font=('Arial', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        plot_frame = ttk.LabelFrame(right_frame, text="Распределение по классам", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_class_plot(plot_frame)
        
        self.setup_styles()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
    def setup_audio_visualization(self, parent):
        self.viz_fig, (self.ax_raw, self.ax_clean) = plt.subplots(2, 1, figsize=(8, 5))
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, parent)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax_raw.set_title("Исходная запись", fontsize=10)
        self.ax_raw.set_ylabel("Амплитуда", fontsize=9)
        self.ax_clean.set_title("После обработки (удаление шумов и молчания)", fontsize=10)
        self.ax_clean.set_ylabel("Амплитуда", fontsize=9)
        self.ax_clean.set_xlabel("Время (сэмплы)", fontsize=9)
        
        self.ax_raw.tick_params(axis='both', which='major', labelsize=8)
        self.ax_clean.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
    def setup_class_plot(self, parent):
        self.class_fig, self.ax_classes = plt.subplots(figsize=(6, 6))
        self.class_canvas = FigureCanvasTkAgg(self.class_fig, parent)
        self.class_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_classes.set_title("Распределение образцов по классам", fontsize=12)
        self.ax_classes.set_xlabel("Слова", fontsize=10)
        self.ax_classes.set_ylabel("Количество записей", fontsize=10)
        
    def update_model_type(self):
        model_type = self.model_var.get()
        self.classifier.set_model_type(model_type)
        
    def update_noise_reduction(self, value):
        self.noise_reduction_level = float(value)
        
    def update_silence_threshold(self, value):
        self.silence_threshold = int(value)
        
    def start_recording_timer(self):
        self.recording_seconds = 3
        self.is_recording = True
        self.update_recording_timer()
        
    def update_recording_timer(self):
        if self.recording_seconds > 0 and self.is_recording:
            minutes = self.recording_seconds // 60
            seconds = self.recording_seconds % 60
            self.record_timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
            self.recording_seconds -= 1
            self.recording_timer = self.root.after(1000, self.update_recording_timer)
        else:
            self.record_timer_label.config(text="00:00")
            self.is_recording = False
            
    def start_testing_timer(self):
        self.testing_seconds = 3
        self.is_testing = True
        self.update_testing_timer()
        
    def update_testing_timer(self):
        if self.testing_seconds > 0 and self.is_testing:
            minutes = self.testing_seconds // 60
            seconds = self.testing_seconds % 60
            self.test_timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
            self.testing_seconds -= 1
            self.testing_timer = self.root.after(1000, self.update_testing_timer)
        else:
            self.test_timer_label.config(text="00:00")
            self.is_testing = False
            
    def stop_timers(self):
        self.is_recording = False
        self.is_testing = False
        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None
        if self.testing_timer:
            self.root.after_cancel(self.testing_timer)
            self.testing_timer = None
            
    def save_audio_to_file(self, audio, label):
        class_dir = os.path.join(self.data_dir, label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.wav"
        filepath = os.path.join(class_dir, filename)
        
        sf.write(filepath, audio, 22050)
        
        return filepath
    
    def load_existing_dataset(self):
        if not os.path.exists(self.data_dir):
            self.status_label.config(text="Папка data не найдена")
            return
            
        self.status_label.config(text="Загрузка датасета...")
        self.progress.start()
        self.root.update()
        
        try:
            self.classifier.features = []
            self.classifier.labels = []
            self.recordings = []
            
            loaded_files = 0
            for class_name in os.listdir(self.data_dir):
                class_dir = os.path.join(self.data_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                    
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        filepath = os.path.join(class_dir, filename)
                        
                        try:
                            audio, sr = librosa.load(filepath, sr=22050)
                            
                            if np.max(np.abs(audio)) > 0:
                                audio = audio / np.max(np.abs(audio))
                            
                            recording_info = {
                                'audio': audio,
                                'filepath': filepath,
                                'label': class_name
                            }
                            self.recordings.append(recording_info)
                            
                            self.classifier.add_sample(audio, class_name)
                            loaded_files += 1
                            
                        except Exception as e:
                            print(f"Ошибка при загрузке файла {filepath}: {e}")
            
            self.update_stats()
            self.status_label.config(text=f"Загружено {loaded_files} записей из датасета")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Ошибка при загрузке датасета: {str(e)}")
            self.status_label.config(text="Ошибка загрузки датасета")
        finally:
            self.progress.stop()
            
    def record_sample(self):
        self.current_word = self.word_entry.get().strip()
        if not self.current_word:
            messagebox.showerror("Ошибка", "Введите слово перед записью")
            return
            
        self.status_label.config(text="Записываю... говорите сейчас!")
        self.record_btn.config(state='disabled')
        self.progress.start()
        self.start_recording_timer()
        self.root.update()
        
        threading.Thread(target=self.record_audio_thread, daemon=True).start()
            
    def record_audio_thread(self):
        try:
            recording = sd.rec(int(3 * 22050), samplerate=22050, channels=1, blocking=True)
            audio = recording.flatten()
            
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            filepath = self.save_audio_to_file(audio, self.current_word)
            
            self.root.after(0, self.process_recorded_audio, audio, filepath)
            
        except Exception as e:
            self.root.after(0, self.record_error, str(e))
            
    def process_recorded_audio(self, audio, filepath):
        recording_info = {
            'audio': audio,
            'filepath': filepath,
            'label': self.current_word
        }
        self.recordings.append(recording_info)
        
        self.visualize_audio(audio)
        
        self.classifier.add_sample(audio, self.current_word)
        self.update_stats()
        self.status_label.config(text=f"Запись '{self.current_word}' завершена и сохранена")
        
        self.progress.stop()
        self.record_btn.config(state='normal')
        self.stop_timers()
        self.record_timer_label.config(text="00:03")
        
    def record_error(self, error_msg):
        messagebox.showerror("Ошибка записи", f"Ошибка: {error_msg}")
        self.progress.stop()
        self.record_btn.config(state='normal')
        self.stop_timers()
        self.record_timer_label.config(text="00:03")
            
    def visualize_audio(self, audio):
        self.ax_raw.clear()
        self.ax_clean.clear()
        
        time_axis = np.arange(len(audio))
        self.ax_raw.plot(time_axis, audio, color='blue', alpha=0.7, linewidth=0.8)
        self.ax_raw.set_title(f"Исходная запись (сэмплов: {len(audio)})", fontsize=10)
        self.ax_raw.set_ylabel("Амплитуда", fontsize=9)
        self.ax_raw.grid(True, alpha=0.3)
        self.ax_raw.tick_params(axis='both', which='major', labelsize=8)
        
        audio_clean = self.classifier.remove_silence_advanced(audio, 22050, self.silence_threshold)
        audio_clean = self.classifier.apply_noise_reduction(audio_clean, 22050, self.noise_reduction_level)
        
        time_axis_clean = np.arange(len(audio_clean))
        self.ax_clean.plot(time_axis_clean, audio_clean, color='green', alpha=0.7, linewidth=0.8)
        self.ax_clean.set_title(f"После обработки (сэмплов: {len(audio_clean)})", fontsize=10)
        self.ax_clean.set_ylabel("Амплитуда", fontsize=9)
        self.ax_clean.set_xlabel("Время (сэмплы)", fontsize=9)
        self.ax_clean.grid(True, alpha=0.3)
        self.ax_clean.tick_params(axis='both', which='major', labelsize=8)
        
        self.viz_canvas.draw()
            
    def train_model(self):
        if len(self.classifier.features) < 6:
            messagebox.showerror("Ошибка", "Нужно как минимум 6 записей для обучения (минимум 2 класса по 3 записи)")
            return
            
        try:
            self.update_model_type()
            
            self.status_label.config(text="Обучение модели...")
            self.train_btn.config(state='disabled')
            self.progress.start()
            self.root.update()
            
            start_time = time.time()
            metrics = self.classifier.train(
                use_cross_validation=self.use_cross_validation.get(),
                cv_folds=self.cv_folds.get()
            )
            training_time = time.time() - start_time
            
            self.update_stats()
            
            model_name = self.get_model_name(self.classifier.model_type)
            
            self.show_metrics_window(model_name, metrics, training_time)
            
            self.status_label.config(text=f"Модель {model_name} обучена за {training_time:.1f} сек")
            
        except Exception as e:
            messagebox.showerror("Ошибка обучения", f"Ошибка: {str(e)}")
        finally:
            self.progress.stop()
            self.train_btn.config(state='normal')
    
    def get_model_name(self, model_type):
        names = {
            "random_forest": "Random Forest",
            "catboost": "CatBoost",
            "logistic_regression": "Logistic Regression"
        }
        return names.get(model_type, model_type)
    
    def format_metrics(self, metrics):
        standard_metrics = metrics['standard_metrics']
        cv_metrics = metrics.get('cv_metrics')
        
        text = f"Accuracy: {standard_metrics['accuracy']:.2%}\n"
        text += f"Precision: {standard_metrics['precision']:.2%}\n"
        text += f"Recall: {standard_metrics['recall']:.2%}\n"
        text += f"F1-Score: {standard_metrics['f1']:.2%}\n"
        
        if cv_metrics:
            text += f"\nКросс-валидация ({self.cv_folds.get()} фолдов):\n"
            text += f"Accuracy: {cv_metrics['accuracy']['mean']:.2%} (±{cv_metrics['accuracy']['std']:.2%})\n"
            text += f"Precision: {cv_metrics['precision']['mean']:.2%} (±{cv_metrics['precision']['std']:.2%})\n"
            text += f"Recall: {cv_metrics['recall']['mean']:.2%} (±{cv_metrics['recall']['std']:.2%})\n"
            text += f"F1-Score: {cv_metrics['f1']['mean']:.2%} (±{cv_metrics['f1']['std']:.2%})\n"
        
        return text
    
    def show_metrics_window(self, model_name, metrics, training_time):
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title(f"Метрики классификации - {model_name}")
        metrics_window.geometry("600x500")
        metrics_window.transient(self.root)
        metrics_window.grab_set()
        
        metrics_window.update_idletasks()
        x = (self.root.winfo_x() + (self.root.winfo_width() - metrics_window.winfo_width()) // 2)
        y = (self.root.winfo_y() + (self.root.winfo_height() - metrics_window.winfo_height()) // 2)
        metrics_window.geometry(f"+{x}+{y}")
        
        main_frame = ttk.LabelFrame(metrics_window, text="Основные метрики", padding=10)
        main_frame.pack(fill=tk.X, padx=10, pady=5)
        
        metrics_text = (f"Модель: {model_name}\n"
                       f"Время обучения: {training_time:.2f} сек\n"
                       f"Использована кросс-валидация: {'Да' if self.use_cross_validation.get() else 'Нет'}\n\n")
        
        ttk.Label(main_frame, text=metrics_text, font=('Arial', 10)).pack(anchor=tk.W)
        
        test_frame = ttk.LabelFrame(metrics_window, text="Метрики на тестовой выборке", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=5)
        
        standard_metrics = metrics['standard_metrics']
        test_text = (f"Accuracy: {standard_metrics['accuracy']:.2%}\n"
                    f"Precision: {standard_metrics['precision']:.2%}\n"
                    f"Recall: {standard_metrics['recall']:.2%}\n"
                    f"F1-Score: {standard_metrics['f1']:.2%}")
        
        ttk.Label(test_frame, text=test_text, font=('Arial', 10)).pack(anchor=tk.W)
        
        if self.use_cross_validation.get() and metrics.get('cv_metrics'):
            cv_frame = ttk.LabelFrame(metrics_window, text=f"Кросс-валидация ({self.cv_folds.get()} фолдов)", padding=10)
            cv_frame.pack(fill=tk.X, padx=10, pady=5)
            
            cv_metrics = metrics['cv_metrics']
            cv_text = (f"Accuracy: {cv_metrics['accuracy']['mean']:.2%} (±{cv_metrics['accuracy']['std']:.2%})\n"
                      f"Precision: {cv_metrics['precision']['mean']:.2%} (±{cv_metrics['precision']['std']:.2%})\n"
                      f"Recall: {cv_metrics['recall']['mean']:.2%} (±{cv_metrics['recall']['std']:.2%})\n"
                      f"F1-Score: {cv_metrics['f1']['mean']:.2%} (±{cv_metrics['f1']['std']:.2%})")
            
            ttk.Label(cv_frame, text=cv_text, font=('Arial', 10)).pack(anchor=tk.W)
        
        report_frame = ttk.LabelFrame(metrics_window, text="Детальный отчет", padding=10)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        report_text = tk.Text(report_frame, wrap=tk.WORD, font=('Arial', 9))
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=report_text.yview)
        report_text.configure(yscrollcommand=scrollbar.set)
        
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        report_text.insert(tk.END, standard_metrics['classification_report'])
        report_text.config(state=tk.DISABLED)
        
        ttk.Button(metrics_window, text="Закрыть", command=metrics_window.destroy).pack(pady=10)
            
    def test_recording(self):
        if self.classifier.model is None:
            messagebox.showerror("Ошибка", "Сначала обучите модель")
            return
            
        self.status_label.config(text="Записываю тестовый образец... говорите сейчас!")
        self.test_btn.config(state='disabled')
        self.progress.start()
        self.start_testing_timer()
        self.root.update()
        
        threading.Thread(target=self.test_audio_thread, daemon=True).start()
    
    def test_audio_thread(self):
        try:
            recording = sd.rec(int(3 * 22050), samplerate=22050, channels=1, blocking=True)
            audio = recording.flatten()
            
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            self.root.after(0, self.process_test_audio, audio)
            
        except Exception as e:
            self.root.after(0, self.test_error, str(e))
            
    def process_test_audio(self, audio):
        self.visualize_audio(audio)
        
        prediction = self.classifier.predict(audio)
        self.status_label.config(text=f"Результат: {prediction}")
        
        model_name = self.get_model_name(self.classifier.model_type)
        messagebox.showinfo("Классификация", 
                          f"Распознано слово: '{prediction}'\n(Использована модель: {model_name})")
        
        self.progress.stop()
        self.test_btn.config(state='normal')
        self.stop_timers()
        self.test_timer_label.config(text="00:03")
        
    def test_error(self, error_msg):
        messagebox.showerror("Ошибка", f"Ошибка: {error_msg}")
        self.progress.stop()
        self.test_btn.config(state='normal')
        self.stop_timers()
        self.test_timer_label.config(text="00:03")
    
    def delete_last_recording(self):
        if not self.classifier.labels:
            messagebox.showinfo("Информация", "Нет записей для удаления")
            return
            
        last_recording = self.recordings[-1] if self.recordings else None
        last_label = self.classifier.remove_last_sample()
        
        if self.recordings:
            if last_recording and os.path.exists(last_recording['filepath']):
                try:
                    os.remove(last_recording['filepath'])
                    class_dir = os.path.join(self.data_dir, last_label)
                    if os.path.exists(class_dir) and not os.listdir(class_dir):
                        os.rmdir(class_dir)
                except Exception as e:
                    print(f"Ошибка при удалении файла: {e}")
            
            self.recordings.pop()
            
        self.update_stats()
        self.status_label.config(text=f"Удалена последняя запись слова '{last_label}'")
        
        if not self.recordings:
            self.ax_raw.clear()
            self.ax_clean.clear()
            self.ax_raw.set_title("Исходная запись", fontsize=10)
            self.ax_raw.set_ylabel("Амплитуда", fontsize=9)
            self.ax_clean.set_title("После обработки (удаление шумов и молчания)", fontsize=10)
            self.ax_clean.set_ylabel("Амплитуда", fontsize=9)
            self.ax_clean.set_xlabel("Время (сэмплы)", fontsize=9)
            self.viz_canvas.draw()
    
    def clear_data(self):
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить все данные?"):
            self.classifier.features = []
            self.classifier.labels = []
            self.classifier.model = None
            self.classifier.metrics = {}
            self.classifier.cv_metrics = {}
            self.recordings = []
            self.update_stats()
            self.status_label.config(text="Все данные очищены")
            
            self.ax_raw.clear()
            self.ax_clean.clear()
            self.ax_raw.set_title("Исходная запись", fontsize=10)
            self.ax_raw.set_ylabel("Амплитуда", fontsize=9)
            self.ax_clean.set_title("После обработки (удаление шумов и молчания)", fontsize=10)
            self.ax_clean.set_ylabel("Амплитуда", fontsize=9)
            self.ax_clean.set_xlabel("Время (сэмплы)", fontsize=9)
            self.viz_canvas.draw()
            
    def update_stats(self):
        stats = "СТАТИСТИКА КОЛЛЕКЦИИ:\n" + "="*40 + "\n"
        counts = {}
        for label in self.classifier.labels:
            counts[label] = counts.get(label, 0) + 1
            
        for word, count in counts.items():
            stats += f"• {word}: {count} записей\n"
            
        stats += "="*40 + "\n"
        stats += f"Всего записей: {len(self.classifier.labels)}\n"
        stats += f"Уникальных слов: {len(counts)}\n"
        
        model_name = self.get_model_name(self.classifier.model_type)
        stats += f"Выбранная модель: {model_name}\n"
        stats += f"Кросс-валидация: {'Включена' if self.use_cross_validation.get() else 'Выключена'}\n"
        
        if self.classifier.model is not None:
            stats += f"\nМодель обучена ✓\n"
            stats += f"Классы: {', '.join(self.classifier.classes)}\n"
            
            if self.classifier.metrics:
                stats += f"\nМетрики классификации:\n"
                stats += f"Accuracy: {self.classifier.metrics['accuracy']:.2%}\n"
                stats += f"Precision: {self.classifier.metrics['precision']:.2%}\n"
                stats += f"Recall: {self.classifier.metrics['recall']:.2%}\n"
                stats += f"F1-Score: {self.classifier.metrics['f1']:.2%}\n"
                
                if self.classifier.cv_metrics:
                    stats += f"\nКросс-валидация:\n"
                    stats += f"Accuracy: {self.classifier.cv_metrics['accuracy']['mean']:.2%} (±{self.classifier.cv_metrics['accuracy']['std']:.2%})\n"
                    stats += f"F1-Score: {self.classifier.cv_metrics['f1']['mean']:.2%} (±{self.classifier.cv_metrics['f1']['std']:.2%})\n"
        else:
            stats += f"\nМодель не обучена\n"
            if len(self.classifier.labels) >= 6:
                stats += f"Готово к обучению!\n"
            else:
                stats += f"Нужно больше записей\n"
            
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        
        self.ax_classes.clear()
        if counts:
            words = list(counts.keys())
            values = list(counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(words)))
            bars = self.ax_classes.bar(words, values, color=colors)
            self.ax_classes.set_title("Распределение образцов по классам", fontsize=12)
            self.ax_classes.set_xlabel("Слова", fontsize=10)
            self.ax_classes.set_ylabel("Количество записей", fontsize=10)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.ax_classes.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{value}', ha='center', va='bottom', fontsize=9)
            
            plt.setp(self.ax_classes.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            self.class_fig.tight_layout()
            self.class_canvas.draw()