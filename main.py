import sys
import os
import cv2
import numpy as np
import yaml
import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QLabel, QVBoxLayout, 
    QHBoxLayout, QWidget, QPushButton, QFileDialog, QStatusBar, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from recognizer import RSLRecognizer

class RSLRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознаватель русского жестового языка")
        self.setGeometry(100, 100, 1000, 700)
        
        # Основные компоненты
        self.capture = None
        self.recognizer = None
        self.tensors_list = []
        self.prediction_list = ["---"]
        self.frame_counter = 0
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
        self.statusBar().showMessage("Готов к работе")
        
    def _init_ui(self):
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Дисплей для видео
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)
        
        # Панель выбора модели и конфигурации
        controls_layout = QHBoxLayout()
        
        # Выбор модели
        model_layout = QVBoxLayout()
        model_label = QLabel("Модель:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.available_models)
        
        self.model_path_button = QPushButton("Выбрать файл модели...")
        self.model_path_button.clicked.connect(self.browse_model)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addWidget(self.model_path_button)
        controls_layout.addLayout(model_layout)
        
        # Выбор конфигурации
        config_layout = QVBoxLayout()
        config_label = QLabel("Конфигурация:")
        self.config_selector = QComboBox()
        self.config_selector.addItems(self.available_configs)
        
        self.config_path_button = QPushButton("Выбрать файл конфигурации...")
        self.config_path_button.clicked.connect(self.browse_config)
        
        config_layout.addWidget(config_label)
        config_layout.addWidget(self.config_selector)
        config_layout.addWidget(self.config_path_button)
        controls_layout.addLayout(config_layout)
        
        # Кнопки управления
        control_buttons_layout = QVBoxLayout()
        self.start_button = QPushButton("Запустить распознавание")
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        
        # Кнопка сохранения результатов
        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        # Кнопка записи видео
        self.record_button = QPushButton("Записать видео")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        
        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.stop_button)
        control_buttons_layout.addWidget(self.save_button)
        control_buttons_layout.addWidget(self.record_button)
        controls_layout.addLayout(control_buttons_layout)
        
        main_layout.addLayout(controls_layout)
        
        # Отображение результатов
        result_layout = QVBoxLayout()
        result_label = QLabel("Распознанные жесты:")
        self.result_display = QLabel("---")
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        
        result_layout.addWidget(result_label)
        result_layout.addWidget(self.result_display)
        main_layout.addLayout(result_layout)
        
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
            self.statusBar().showMessage("Ошибка: модели не найдены")
            return
        
        model_path = os.path.join(self.models_dir, model_name)
        
        # Получение выбранной конфигурации
        config_name = self.config_selector.currentText()
        config = {}
        if config_name != "Конфигурации не найдены":
            config_path = os.path.join(self.configs_dir, config_name)
            config = self.load_config(config_path)
        
        # Инициализация распознавателя
        classes_path = "classes.json"  # Путь к файлу с классами жестов
        self.recognizer = RSLRecognizer(model_path, config, classes_path)
        if not self.recognizer.session:
            self.statusBar().showMessage("Ошибка загрузки модели")
            return
        
        # Инициализация камеры
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.statusBar().showMessage("Ошибка: невозможно открыть камеру")
            return
        
        # Сброс списков
        self.tensors_list = []
        self.prediction_list = ["---"]
        self.frame_counter = 0
        
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
        self.statusBar().showMessage("Распознавание запущено")
    
    def stop_recognition(self):
        """Останавливает процесс распознавания"""
        # Останавливаем запись видео, если она идет
        if self.is_recording:
            self.toggle_recording()
        
        # Остановка обработки
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Обновление интерфейса
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)  # Теперь можно сохранить результаты
        self.record_button.setEnabled(False)  # Нельзя записывать видео без распознавания
        self.model_selector.setEnabled(True)
        self.config_selector.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.config_path_button.setEnabled(True)
        
        self.statusBar().showMessage("Распознавание остановлено")
    
    def save_results(self):
        """Сохраняет распознанные жесты в текстовый файл"""
        if not self.prediction_list or len(self.prediction_list) <= 1:  # Только "---"
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для сохранения")
            return
        
        # Формируем имя файла с текущей датой и временем
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = self.model_selector.currentText().split('.')[0]  # Убираем расширение
        default_filename = f"results/РЖЯ_распознавание_{model_name}_{timestamp}.txt"
        
        # Диалог выбора места сохранения
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", default_filename, 
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if not file_path:
            return  # Пользователь отменил сохранение
        
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Сохраняем результаты в файл
            with open(file_path, 'w', encoding='utf-8') as f:
                # Заголовок
                f.write(f"# Результаты распознавания русского жестового языка\n")
                f.write(f"# Дата и время: {timestamp}\n")
                f.write(f"# Модель: {self.model_selector.currentText()}\n")
                f.write(f"# Конфигурация: {self.config_selector.currentText()}\n")
                f.write("\n")
                
                # Результаты
                f.write("## Распознанные жесты:\n")
                for i, gesture in enumerate(self.prediction_list):
                    if gesture != "---":  # Пропускаем начальное значение
                        f.write(f"{i}. {gesture}\n")
                
                f.write("\n")
                f.write("## Последовательность жестов:\n")
                f.write(" ".join([g for g in self.prediction_list if g != "---"]))
            
            self.statusBar().showMessage(f"Результаты сохранены в {file_path}")
            
            # Показываем сообщение об успешном сохранении
            QMessageBox.information(self, "Успех", f"Результаты успешно сохранены в\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результаты: {e}")
            self.statusBar().showMessage(f"Ошибка сохранения: {e}")
    
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
    
    def update_frame(self):
        """Обновляет кадр с камеры и выполняет распознавание"""
        if not self.capture or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        # Добавляем на кадр распознанный жест
        frame_with_text = frame.copy()
        if self.prediction_list and self.prediction_list[-1] != "---":
            # Добавляем текст с распознанным жестом
            text = self.prediction_list[-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)  # Зеленый цвет
            text_position = (20, 50)  # Позиция в левом верхнем углу
            
            # Добавляем фон для текста для лучшей читаемости
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(
                frame_with_text, 
                (text_position[0] - 10, text_position[1] - text_size[1] - 10),
                (text_position[0] + text_size[0] + 10, text_position[1] + 10),
                (0, 0, 0),  # Черный фон
                -1  # Заполненный прямоугольник
            )
            
            # Добавляем текст
            cv2.putText(
                frame_with_text, text, text_position, font, font_scale, 
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
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), 
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
                    
                    # Получаем результат распознавания
                    gloss, confidence = self.recognizer.predict(input_tensor)
                    
                    # Обрабатываем корректно результат распознавания
                    if gloss is not None and confidence > 0.2:  # Минимальный порог уверенности
                        if gloss != self.prediction_list[-1]:
                            self.prediction_list.append(gloss)
                            # Ограничиваем список предсказаний последними 5 элементами
                            if len(self.prediction_list) > 5:
                                self.prediction_list = self.prediction_list[-5:]
                            
                            # Активируем кнопку сохранения, если есть результаты
                            if not self.save_button.isEnabled() and len([g for g in self.prediction_list if g != "---"]) > 0:
                                self.save_button.setEnabled(True)
                            
                            # Обновление отображения результатов
                            text = "  ".join(self.prediction_list)
                            self.result_display.setText(text)
                            
                    # Очистка буфера кадров
                    self.tensors_list = []
                    
                except Exception as e:
                    self.tensors_list = []  # Очищаем буфер в случае ошибки
                    print(f"Ошибка при обработке кадров: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RSLRecognitionApp()
    window.show()
    sys.exit(app.exec_()) 