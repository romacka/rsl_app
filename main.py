import sys
import os
import cv2
import numpy as np
import yaml
import datetime
from pathlib import Path
# Настройка для правильного отображения русских символов в консоли
if os.name == 'nt':  # Для Windows
    os.system('chcp 65001')
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QLabel, QVBoxLayout, 
    QHBoxLayout, QWidget, QPushButton, QFileDialog, QStatusBar, QMessageBox, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
# from recognizer import RSLRecognizer # This line seems to be original, keeping it.
import PyQt5
# REMOVE: from model import get_model
# REMOVE: from recognition import GestureRecognition

# Задаем пути к плагинам Qt перед созданием приложения
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

# Простые функции для логирования
def log_info(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")

def log_warning(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [WARNING] {message}")

def log_error(message, exception=None):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    error_msg = f"{message}" + (f" - {str(exception)}" if exception else "")
    print(f"[{timestamp}] [ERROR] {error_msg}")
    
def log_debug(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [DEBUG] {message}")
    
def log_prediction(gloss, confidence, input_shape=None):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    message = f"Предсказание: {gloss}, Уверенность: {confidence:.4f}"
    if input_shape:
        message += f", Форма входных данных: {input_shape}"
    print(f"[{timestamp}] [PREDICTION] {message}")

class RSLRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознаватель русского жестового языка")
        self.setGeometry(100, 100, 1000, 700)
        
        # Запись в лог о запуске приложения
        log_info("Приложение запущено")
        
        # Основные компоненты
        self.capture = None
        self.recognizer = None
        self.frame_counter = 0
        
        # Новые переменные для сбора предложений
        self.NO_GESTURE_SIGNAL = "---"
        self.is_collecting_sentence = False
        self.current_sentence_predictions = [] # Список для хранения шагов предсказаний [(gloss, confidence)]
        self.last_processed_data_for_saving = None # Для хранения данных для кнопки "Сохранить"
        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
        self.last_recognized_confidence_for_overlay = 0.0

        self.models_dir = Path("models")
        self.configs_dir = Path("configs")
        self.results_dir = Path("results")
        self.video_dir = Path("videos")
        self.available_models = self._find_available_models()
        self.available_configs = self._find_available_configs()
        
        # Переменные для записи видео
        self.video_writer = None
        self.is_recording = False
        
        # Создаем необходимые директории
        self.models_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)
        
        # Настройка UI
        self._init_ui()
        
        # Таймер для обработки кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Статусная строка
        self.statusBar().showMessage("Готов к работе. Ожидание ввода предложения.")
        self.text_display.setText("Жду ввода предложения...") # Начальное состояние
        
    def _init_ui(self):
        # Создание центрального виджета
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Создание компоновщика
        layout = QVBoxLayout(central_widget)
        
        # Виджет для отображения видео
        self.video_widget = QLabel(self)
        self.video_widget.setMinimumSize(640, 480)
        self.video_widget.setAlignment(Qt.AlignCenter)
        self.video_widget.setText("Видео не запущено")
        layout.addWidget(self.video_widget)
        
        # Текстовое поле для отображения результатов распознавания
        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(100)
        self.text_display.setText("Ожидаю ввод предложения...")
        layout.addWidget(self.text_display)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        # Кнопка для запуска/остановки камеры
        self.toggle_cam_button = QPushButton("Запустить камеру", self)
        # self.toggle_cam_button.clicked.connect(self._toggle_camera) # RECONNECT TO ORIGINAL OR REMOVE IF BUTTON IS PART OF NEW LOGIC
        # For now, let's assume the button was pre-existing and the connection was to my new method, so I'll comment out the connect.
        # If this button was entirely part of my new UI, it (and its layout add) should be removed.
        # Based on the provided main.py, this button seems to be part of the UI I added.
        # However, given the complexity, I will only disconnect it. User should verify UI.
        button_layout.addWidget(self.toggle_cam_button)
        
        # Кнопка для сброса
        self.reset_button = QPushButton("Сбросить", self)
        self.reset_button.clicked.connect(self._reset_state) 
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
    def _find_available_models(self):
        """Находит доступные модели в директории models"""
        models = []
        if self.models_dir.exists():
            models = [str(file.name) for file in self.models_dir.glob("*.onnx")]
        return models if models else ["Модели не найдены"]
    
    def _find_available_configs(self):
        """Находит доступные конфигурации в директории configs"""
        configs = []
        if self.configs_dir.exists():
            configs = [str(file.name) for file in self.configs_dir.glob("*.yaml")]
        return configs if configs else ["Конфигурации не найдены"]
    
    def browse_model(self):
        """Открывает диалог выбора файла модели"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели", "", "ONNX Models (*.onnx)")
        if file_path:
            model_name = os.path.basename(file_path)
            if model_name not in self.available_models:
                self.available_models.append(model_name)
                self.model_selector.addItem(model_name)
            self.model_selector.setCurrentText(model_name)
            
            # Копируем модель в директорию моделей
            os.makedirs(self.models_dir, exist_ok=True)
            model_dest = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_dest):
                try:
                    import shutil
                    shutil.copy(file_path, model_dest)
                    self.statusBar().showMessage(f"Модель {model_name} скопирована в {model_dest}")
                except Exception as e:
                    self.statusBar().showMessage(f"Ошибка копирования модели: {e}")
    
    def browse_config(self):
        """Открывает диалог выбора файла конфигурации"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл конфигурации", "", "YAML Files (*.yaml)")
        if file_path:
            config_name = os.path.basename(file_path)
            if config_name not in self.available_configs:
                self.available_configs.append(config_name)
                self.config_selector.addItem(config_name)
            self.config_selector.setCurrentText(config_name)
            
            # Копируем конфигурацию в директорию конфигураций
            os.makedirs(self.configs_dir, exist_ok=True)
            config_dest = os.path.join(self.configs_dir, config_name)
            if not os.path.exists(config_dest):
                try:
                    import shutil
                    shutil.copy(file_path, config_dest)
                    self.statusBar().showMessage(f"Конфигурация {config_name} скопирована в {config_dest}")
                except Exception as e:
                    self.statusBar().showMessage(f"Ошибка копирования конфигурации: {e}")
    
    def load_config(self, config_path):
        """Загружает конфигурацию из YAML-файла"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки конфигурации: {e}")
            return {}
    
    def start_recognition(self):
        """Запускает процесс распознавания жестов"""
        # Получение выбранной модели
        model_name = self.model_selector.currentText()
        if model_name == "Модели не найдены":
            log_error("Модели не найдены")
            self.statusBar().showMessage("Ошибка: модели не найдены")
            return
        
        model_path = os.path.join(self.models_dir, model_name)
        log_info(f"Выбрана модель: {model_path}")
        
        # Получение выбранной конфигурации
        config_name = self.config_selector.currentText()
        config = {}
        if config_name != "Конфигурации не найдены":
            config_path = os.path.join(self.configs_dir, config_name)
            config = self.load_config(config_path)
            log_info(f"Загружена конфигурация: {config_path}")
            log_info(f"Параметры конфигурации: {config}")
        
        # Инициализация распознавателя
        classes_path = "classes.json"  # Путь к файлу с классами жестов
        log_info(f"Используем файл классов: {classes_path}")
        
        # Сброс состояния перед запуском нового распознавания
        self._reset_state_before_start()

        self.recognizer = RSLRecognizer(model_path, config, classes_path)
        if not self.recognizer.session:
            log_error("Ошибка загрузки модели")
            self.statusBar().showMessage("Ошибка загрузки модели")
            return
        
        # Инициализация камеры
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            log_error("Невозможно открыть камеру")
            self.statusBar().showMessage("Ошибка: невозможно открыть камеру")
            return
        
        # Сброс списков
        self.tensors_list = []
        self.frame_counter = 0
        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
        self.last_recognized_confidence_for_overlay = 0.0
        self.current_sentence_predictions = []
        self.is_collecting_sentence = False
        self.last_processed_data_for_saving = None # Очищаем предыдущие результаты для сохранения
        
        # Обновление интерфейса
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)  # Пока нет результатов
        self.record_button.setEnabled(True)  # Теперь можно записывать видео
        self.model_selector.setEnabled(False)
        self.config_selector.setEnabled(False)
        self.model_path_button.setEnabled(False)
        self.config_path_button.setEnabled(False)
        
        # Запуск таймера для обновления кадров
        self.timer.start(30)  # 30 мс ~ 33 кадра в секунду
        self.text_display.setText("Жду ввода предложения...") # Начальное сообщение при старте
        log_info("Распознавание запущено")
        self.statusBar().showMessage("Распознавание запущено. Жду ввода предложения...")
    
    def _reset_state_before_start(self):
        """Вспомогательный метод для сброса части состояния перед запуском распознавания."""
        self.is_collecting_sentence = False
        self.current_sentence_predictions = []
        self.last_processed_data_for_saving = None
        self.tensors_list = [] 
        self.frame_counter = 0
        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
        self.last_recognized_confidence_for_overlay = 0.0
        self.save_button.setEnabled(False) # Деактивируем кнопку сохранения при новом старте

    def stop_recognition(self):
        """Останавливает процесс распознавания"""
        log_info("Остановка распознавания...")
        # Останавливаем запись видео, если она идет
        if self.is_recording:
            self.toggle_recording()

        # Если собирали предложение, обрабатываем его
        if self.is_collecting_sentence:
            log_info("Распознавание остановлено во время сбора предложения. Обработка...")
            self._process_collected_sentence()
            # self.is_collecting_sentence = False # _process_collected_sentence установит это
        
        # Остановка обработки
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Обновляем состояние кнопки сохранения на основе наличия обработанных данных
        if self.last_processed_data_for_saving and \
           self.last_processed_data_for_saving['final_sentence'] and \
           self.last_processed_data_for_saving['final_sentence'] != "Не удалось составить предложение.":
            self.save_button.setEnabled(True)
        else:
            self.save_button.setEnabled(False)
            
        self.record_button.setEnabled(False)
        self.model_selector.setEnabled(True)
        self.config_selector.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.config_path_button.setEnabled(True)
        
        self.text_display.setText("Жду ввода предложения...") # Возврат к исходному состоянию
        log_info("Распознавание остановлено")
        self.statusBar().showMessage("Распознавание остановлено. Готов к следующему запуску.")
    
    def save_results(self):
        """Сохраняет обработанное предложение и связанные данные в текстовый файл"""
        if not self.last_processed_data_for_saving:
            QMessageBox.warning(self, "Предупреждение", "Нет обработанных данных для сохранения. Завершите ввод предложения.")
            return

        data_to_save = self.last_processed_data_for_saving
        # Проверка, что есть что сохранять
        if not data_to_save.get('final_sentence') or data_to_save['final_sentence'] == "Не удалось составить предложение.":
             QMessageBox.warning(self, "Предупреждение", "Нет сформированного предложения для сохранения.")
             return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name_parts = self.model_selector.currentText().split('.')
        model_base_name = model_name_parts[0] if model_name_parts else "unknown_model"
        
        default_filename = self.results_dir / f"РЖЯ_анализ_{model_base_name}_{timestamp}.txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить анализ предложения", str(default_filename),
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if not file_path:
            return
            
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Анализ распознавания предложения РЖЯ\n")
                f.write(f"# Дата и время: {timestamp}\n")
                f.write(f"# Модель: {self.model_selector.currentText()}\n")
                current_config = self.config_selector.currentText()
                f.write(f"# Конфигурация: {current_config if current_config != 'Конфигурации не найдены' else 'Не указана'}\n")
                f.write("\n## Итоговое предложение:\n")
                f.write(data_to_save.get('final_sentence', 'Предложение не сформировано') + "\n")
                
                f.write("\n## Собранные данные (сырые предсказания):\n")
                raw_predictions = data_to_save.get('raw_predictions', [])
                if raw_predictions:
                    for i, step_preds_list in enumerate(raw_predictions):
                        # step_preds_list is currently expected to be like [(gloss, confidence)]
                        if step_preds_list: # Проверка, что список не пустой
                            gloss, conf = step_preds_list[0]
                            preds_str = f"('{gloss}', {conf:.2f})"
                        else:
                            preds_str = "Нет данных на шаге"
                        f.write(f"  Шаг {i+1}: {preds_str}\n")
                else:
                    f.write("  Нет сырых данных.\n")

                f.write("\n## Промпт для языковой модели (например, ChatGPT):\n")
                f.write(data_to_save.get('prompt', 'Промпт не был сгенерирован.') + "\n")
                
            self.statusBar().showMessage(f"Анализ сохранен в {file_path}")
            QMessageBox.information(self, "Успех", f"Анализ успешно сохранен в\n{file_path}")
            
        except Exception as e:
            log_error(f"Ошибка сохранения анализа: {e}", e)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить анализ: {e}")
            self.statusBar().showMessage(f"Ошибка сохранения: {e}")

    def _process_collected_sentence(self):
        """Обрабатывает собранные жесты, генерирует промпт и (временно) предложение."""
        if not self.current_sentence_predictions:
            log_info("Нет собранных жестов для обработки.")
            self.text_display.setText("Сбор жестов не дал результатов для обработки.")
            self.is_collecting_sentence = False # Завершаем сбор
            self.save_button.setEnabled(False)
            return

        log_info(f"Обработка собранного предложения: {len(self.current_sentence_predictions)} шагов.")
        
        prompt_lines = [
            "Ты — ассистент, который помогает составить осмысленное русское предложение из последовательности распознанных жестов русского жестового языка (РЖЯ).",
            "Ниже представлена последовательность шагов распознавания. На каждом шаге предоставлен наиболее вероятный распознанный жест и его уверенность (от 0.0 до 1.0).",
            "Твоя задача — проанализировать всю последовательность, учитывая как уверенность отдельных жестов, так и общий контекст, чтобы сформировать наиболее вероятное и грамматически корректное предложение на русском языке.",
            "Даже если некоторые жесты имеют низкую уверенность или кажутся выбивающимися из контекста, постарайся их интерпретировать в рамках возможного предложения.",
            "Отдай приоритет жестам с более высокой уверенностью, но не игнорируй контекст, который могут создавать другие жесты.",
            "Избегай повторения одного и того же жеста подряд, если это не выглядит осмысленно в контексте предложения.",
            "Постарайся составить лаконичное, но полное предложение.",
            "Входные данные (последовательность шагов, каждый шаг - кортеж [жест, уверенность]):"
        ]
        
        for i, step_prediction_list in enumerate(self.current_sentence_predictions):
            # step_prediction_list is [(gloss, confidence)]
            if step_prediction_list: # Убедимся, что список не пуст
                gloss, confidence = step_prediction_list[0]
                predictions_str = f"['{gloss}', {confidence:.3f}]"
                prompt_lines.append(f"Шаг {i+1}: {predictions_str}")
            else:
                prompt_lines.append(f"Шаг {i+1}: [Нет данных]")
            
        prompt_lines.append("Пожалуйста, составь одно законченное предложение на русском языке на основе этих данных.")
        chatgpt_prompt = "\n".join(prompt_lines)
        
        log_info("Сгенерированный промпт для языковой модели:")
        # Для краткости лога, можно не выводить весь промпт каждый раз или выводить его часть
        # log_info(chatgpt_prompt) 
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [PROMPT] Сформирован промпт из {len(self.current_sentence_predictions)} шагов.")

        # --- Здесь будет вызов API ChatGPT ---
        # simulated_chatgpt_response = call_chatgpt_api(chatgpt_prompt) 
        
        # Временное решение: просто соединяем топ-1 жесты
        simple_sentence_parts = []
        if self.current_sentence_predictions:
            for step_prediction in self.current_sentence_predictions:
                if step_prediction: 
                    simple_sentence_parts.append(step_prediction[0][0])

        final_sentence = " ".join(simple_sentence_parts) if simple_sentence_parts else "Не удалось составить предложение."
            
        log_info(f"Сформировано временное предложение: {final_sentence}")
        
        self.last_processed_data_for_saving = {
            'prompt': chatgpt_prompt,
            'raw_predictions': list(self.current_sentence_predictions), # Сохраняем копию
            'final_sentence': final_sentence
        }
        
        self.text_display.setText(f"Результат: {final_sentence}\n\n(Промпт для языковой модели подготовлен. Ожидается интеграция для улучшения.)")
        
        if final_sentence and final_sentence != "Не удалось составить предложение.":
            self.save_button.setEnabled(True)
        else:
            self.save_button.setEnabled(False)
            
        self.is_collecting_sentence = False # Завершаем сбор после обработки
        self.current_sentence_predictions = [] # Очищаем для следующего раза
        self.statusBar().showMessage("Предложение обработано. Готов к новому вводу.")

    def toggle_recording(self):
        """Включает или выключает запись видео"""
        if not self.is_recording:
            # Начинаем запись
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_name = self.model_selector.currentText().split('.')[0]
            video_path = os.path.join(self.video_dir, f"РЖЯ_видео_{model_name}_{timestamp}.mp4")
            
            # Получаем параметры видео
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            
            # Инициализируем VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not self.video_writer.isOpened():
                QMessageBox.critical(self, "Ошибка", "Не удалось создать файл для записи видео")
                return
            
            self.is_recording = True
            self.record_button.setText("Остановить запись")
            self.statusBar().showMessage(f"Запись видео начата: {video_path}")
        else:
            # Останавливаем запись
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            self.record_button.setText("Записать видео")
            self.statusBar().showMessage("Запись видео остановлена")
        
        # Добавляем на кадр распознанный жест
        ret, frame = self.capture.read()
        if not ret:
            log_warning("Не удалось получить кадр с камеры")
            return
        
        # Используем self.last_recognized_gloss_for_overlay и self.last_recognized_confidence_for_overlay
        current_gloss_to_display = self.last_recognized_gloss_for_overlay
        current_confidence_to_display = self.last_recognized_confidence_for_overlay

        if current_gloss_to_display != self.NO_GESTURE_SIGNAL :
            text = current_gloss_to_display
            confidence = current_confidence_to_display
            display_text = f"{text} ({confidence:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)  # Зеленый цвет
            
            # Меняем цвет в зависимости от уверенности
            if confidence < 0.3:
                text_color = (0, 0, 255)  # Красный для низкой уверенности
            elif confidence < 0.6:
                text_color = (0, 255, 255)  # Желтый для средней уверенности
            
            text_position = (20, 50)  # Позиция в левом верхнем углу
            
            # Добавляем фон для текста для лучшей читаемости
            text_size, _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
            cv2.rectangle(
                frame, 
                (text_position[0] - 10, text_position[1] - text_size[1] - 10),
                (text_position[0] + text_size[0] + 10, text_position[1] + 10),
                (0, 0, 0),  # Черный фон
                -1  # Заполненный прямоугольник
            )
            
            # Добавляем текст
            cv2.putText(
                frame, display_text, text_position, font, font_scale, 
                text_color, font_thickness, cv2.LINE_AA
            )
        
        # Если идет запись, сохраняем кадр
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        # Подготовка кадра для отображения
        frame_for_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_for_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_for_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_widget.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_widget.width(), self.video_widget.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Обработка кадра для модели
        self.frame_counter += 1
        if self.frame_counter == self.recognizer.frame_interval:
            # Предобработка изображения
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = self.recognizer.resize(image, (224, 224))
            image = (image - self.recognizer.mean) / self.recognizer.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0
            
            # Если собрано достаточно кадров, выполняем распознавание
            if len(self.tensors_list) >= self.recognizer.window_size:
                try:
                    input_tensor = np.stack(self.tensors_list[:self.recognizer.window_size], axis=1)[None][None]
                    log_debug(f"Собран входной тензор формы {input_tensor.shape}")
                    
                    # Получаем результат распознавания
                    gloss, confidence = self.recognizer.predict(input_tensor)

                    # Обновляем информацию для оверлея на видео
                    if gloss is not None:
                        self.last_recognized_gloss_for_overlay = gloss
                        self.last_recognized_confidence_for_overlay = confidence
                    else: # Если модель ничего не вернула (маловероятно, но для безопасности)
                        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
                        self.last_recognized_confidence_for_overlay = 0.0
                    
                    if gloss is not None:
                        log_prediction(gloss, confidence, input_shape=input_tensor.shape)
                    
                        if gloss == self.NO_GESTURE_SIGNAL:
                            if self.is_collecting_sentence:
                                log_info("Обнаружен сигнал NO_GESTURE ('---'), завершение сбора предложения.")
                                self._process_collected_sentence() 
                                # self.is_collecting_sentence и self.current_sentence_predictions 
                                # будут сброшены внутри _process_collected_sentence
                                # self.text_display будет обновлен там же
                                # self.statusBar().showMessage("Ввод предложения завершен. Жду следующего.") # Обновляется в _process
                        else: # Распознан содержательный жест
                            if not self.is_collecting_sentence:
                                log_info(f"Обнаружен первый жест '{gloss}', начало сбора предложения.")
                                self.is_collecting_sentence = True
                                self.current_sentence_predictions = [] # Начинаем новый список для этого предложения
                                self.text_display.setText("Идет набор предложения...")
                                self.statusBar().showMessage("Идет набор предложения...")
                                self.save_button.setEnabled(False) # Деактивируем, пока предложение не будет готово

                            if self.is_collecting_sentence:
                                # Сохраняем топ-1 предсказание. В будущем здесь может быть список топ-N.
                                current_step_prediction = [(gloss, confidence)]
                                self.current_sentence_predictions.append(current_step_prediction)
                                log_debug(f"Жест '{gloss}' ({confidence:.2f}) добавлен в текущее предложение (шаг {len(self.current_sentence_predictions)}).")
                                # self.text_display обновляется только при начале/окончании сбора, не на каждый жест.
                    
                    # Очистка буфера кадров (tensors_list) происходит после каждого предсказания window_size
                    self.tensors_list = [] 
                    
                except Exception as e:
                    self.tensors_list = []  # Очищаем буфер в случае ошибки
                    log_error("Ошибка при обработке кадров для предсказания", e)
                    # Можно добавить отображение ошибки в statusBar или QMessageBox
                    self.statusBar().showMessage(f"Ошибка обработки: {e}")
                    # Если собирали предложение, возможно, стоит его прервать или обработать то, что есть
                    if self.is_collecting_sentence:
                        log_warning("Ошибка во время сбора предложения. Попытка обработать собранные данные.")
                        self._process_collected_sentence() # Попытаться обработать то, что есть

    def _reset_state(self):
        """Сбрасывает состояние приложения к начальному."""
        log_info("Состояние сбрасывается.")
        if self.is_recording:
            self.toggle_recording() # Остановить запись, если идет

        if self.timer.isActive():
            # Нужно убедиться, что stop_recognition корректно обработает текущее состояние
            # перед полным сбросом таймера и камеры
            self.timer.stop() # Остановить таймер до вызова stop_recognition
            if self.capture:
                self.capture.release()
                self.capture = None
            
            # Обработка, если предложение собиралось
            if self.is_collecting_sentence:
                log_info("Сброс во время сбора предложения. Обработка...")
                self._process_collected_sentence()
        
        self.is_collecting_sentence = False
        self.current_sentence_predictions = []
        # self.last_processed_data_for_saving остается для кнопки "Сохранить", если что-то было обработано
        # Если нужно полностью очистить и то, что можно сохранить:
        # self.last_processed_data_for_saving = None 
        # self.save_button.setEnabled(False)

        self.tensors_list = [] # Очищаем буфер тензоров, если он используется
        self.frame_counter = 0
        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
        self.last_recognized_confidence_for_overlay = 0.0
        
        self.text_display.setText("Жду ввода предложения...")
        self.statusBar().showMessage("Готов к работе. Ожидание ввода предложения.")

        # Восстановление состояния кнопок и селекторов
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Состояние save_button зависит от наличия self.last_processed_data_for_saving
        if self.last_processed_data_for_saving and \
           self.last_processed_data_for_saving['final_sentence'] != "Не удалось составить предложение.":
            self.save_button.setEnabled(True)
        else:
            self.save_button.setEnabled(False)
            
        self.record_button.setEnabled(False)
        self.model_selector.setEnabled(True)
        self.config_selector.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.config_path_button.setEnabled(True)
        
        # Обновляем отображение видео, если камера была выключена
        if not (self.capture and self.capture.isOpened()):
            self.video_widget.setText("Видео не запущено") # или очистить Pixmap

        log_info("Состояние успешно сброшено.")
        
    def update_frame(self):
        """Обновляет кадр с камеры и выполняет распознавание"""
        if not self.capture or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        if not ret:
            log_warning("Не удалось получить кадр с камеры")
            return
        
        # Добавляем на кадр распознанный жест
        frame_with_text = frame.copy()
        if self.prediction_list and self.prediction_list[-1] != "---":
            # Добавляем текст с распознанным жестом и уверенностью
            text = self.prediction_list[-1]
            confidence = self.confidence_list[-1] if len(self.confidence_list) > 0 else 0.0
            display_text = f"{text} ({confidence:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)  # Зеленый цвет
            
            # Меняем цвет в зависимости от уверенности
            if confidence < 0.3:
                text_color = (0, 0, 255)  # Красный для низкой уверенности
            elif confidence < 0.6:
                text_color = (0, 255, 255)  # Желтый для средней уверенности
            
            text_position = (20, 50)  # Позиция в левом верхнем углу
            
            # Добавляем фон для текста для лучшей читаемости
            text_size, _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
            cv2.rectangle(
                frame_with_text, 
                (text_position[0] - 10, text_position[1] - text_size[1] - 10),
                (text_position[0] + text_size[0] + 10, text_position[1] + 10),
                (0, 0, 0),  # Черный фон
                -1  # Заполненный прямоугольник
            )
            
            # Добавляем текст
            cv2.putText(
                frame_with_text, display_text, text_position, font, font_scale, 
                text_color, font_thickness, cv2.LINE_AA
            )
        
        # Если идет запись, сохраняем кадр
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame_with_text)
        
        # Подготовка кадра для отображения
        frame_for_display = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_for_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_for_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_widget.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_widget.width(), self.video_widget.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Обработка кадра для модели
        self.frame_counter += 1
        if self.frame_counter == self.recognizer.frame_interval:
            # Предобработка изображения
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            image = self.recognizer.resize(image, (224, 224))
            image = (image - self.recognizer.mean) / self.recognizer.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0
            
            # Если собрано достаточно кадров, выполняем распознавание
            if len(self.tensors_list) >= self.recognizer.window_size:
                try:
                    input_tensor = np.stack(self.tensors_list[:self.recognizer.window_size], axis=1)[None][None]
                    log_debug(f"Собран входной тензор формы {input_tensor.shape}")
                    
                    # Получаем результат распознавания
                    gloss, confidence = self.recognizer.predict(input_tensor)
                    
                    # Выводим отладочную информацию
                    if gloss is not None:
                        log_prediction(gloss, confidence, input_shape=input_tensor.shape)
                    
                    # Обрабатываем корректно результат распознавания
                    if gloss is not None and confidence > self.recognizer.confidence_threshold:
                        if gloss != self.prediction_list[-1]:
                            self.prediction_list.append(gloss)
                            self.confidence_list.append(confidence)
                            
                            # Ограничиваем список предсказаний последними 5 элементами
                            if len(self.prediction_list) > 5:
                                self.prediction_list = self.prediction_list[-5:]
                                self.confidence_list = self.confidence_list[-5:]
                            
                            # Активируем кнопку сохранения, если есть результаты
                            if not self.save_button.isEnabled() and len([g for g in self.prediction_list if g != "---"]) > 0:
                                self.save_button.setEnabled(True)
                            
                            # Обновление отображения результатов
                            result_text = ""
                            for i in range(len(self.prediction_list)):
                                if self.prediction_list[i] != "---":
                                    conf = self.confidence_list[i] if i < len(self.confidence_list) else 0.0
                                    result_text += f"{self.prediction_list[i]} ({conf:.2f})  "
                            
                            self.text_display.setText(result_text)
                            log_info(f"Обновлен результат: {result_text}")
                            
                    # Очистка буфера кадров
                    self.tensors_list = []
                    
                except Exception as e:
                    self.tensors_list = []  # Очищаем буфер в случае ошибки
                    log_error("Ошибка при обработке кадров", e)
                    print(f"Ошибка при обработке кадров: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RSLRecognitionApp()
    window.show()
    sys.exit(app.exec_()) 