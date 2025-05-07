import sys
import os
import cv2
import numpy as np
import yaml
import datetime
from pathlib import Path
from recognizer import RSLRecognizer
from dotenv import load_dotenv
import openai # <--- Раскомментируем openai
# import requests # <--- Комментируем requests
# import json # <--- Комментируем json
import httpx # <--- Добавляем импорт httpx
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
from recognizer import RSLRecognizer 

# Загрузка переменных окружения из .env файла
load_dotenv()

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

# Функция для вызова API ChatGPT с использованием requests
def get_chatgpt_response(prompt_text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log_error("API-ключ OpenAI не найден в переменных окружения.")
        return "Ошибка: API-ключ OpenAI не настроен."

    # openai.api_key = api_key # Устанавливаем ключ для библиотеки openai
    # org_id = os.getenv("OPENAI_ORG_ID") # Если используется ключ проекта, может потребоваться ID организации
    # if org_id:
    #    openai.organization = org_id

    # Используем стандартную модель и эндпоинт Chat Completions
    # Если вы хотите использовать gpt-4.1-nano и другой эндпоинт, их нужно будет указать здесь
    model_to_use = "gpt-4.1-nano" # Оставляем модель, которую вы использовали
    # api_url = "https://api.openai.com/v1/chat/completions" # Не используется с библиотекой openai

    # --- Убираем ВРЕМЕННЫЙ ТЕСТ --- 
    # simple_test_payload = {
    # "model": model_to_use,
    # "messages": [{"role": "user", "content": "Say this is a test!"}],
    # "temperature": 0.7 
    # }
    # payload_to_send = simple_test_payload
    # -------------------------------------------------------------

    messages = [
        {"role": "system", "content": "Ты — полезный ассистент, который помогает составить осмысленное русское предложение из данных распознавания жестов."},
        {"role": "user", "content": prompt_text}
    ]
    
    log_info(f"Подготовка запроса в ChatGPT с использованием библиотеки 'openai' (модель: {model_to_use})...")
    # log_debug(f"Тело запроса (messages) для ChatGPT:\n------ MESSAGES НАЧАЛО ------\n{json.dumps(messages, indent=2, ensure_ascii=False)}\n------ MESSAGES КОНЕЦ ------") # json.dumps здесь не нужен, так как json не импортируется

    try:
        log_info(f"Отправка запроса в ChatGPT (модель: {model_to_use})...")
        
        # --- ДИАГНОСТИКА SSL: Начало ---
        # Попробуем отключить SSL-верификацию для диагностики FileNotFoundError
        # Это НЕ рекомендуется для постоянного использования!
        # Если это поможет, значит проблема в SSL-сертификатах вашего окружения.
        custom_httpx_client = httpx.Client(verify=False)
        client = openai.OpenAI(api_key=api_key, http_client=custom_httpx_client)
        # --- ДИАГНОСТИКА SSL: Конец ---
        
        # Старая инициализация клиента (закомментирована для теста):
        # client = openai.OpenAI(api_key=api_key) 

        # org_id = os.getenv("OPENAI_ORG_ID")
        # if org_id:
        # client.organization = org_id # Эта строка была закомментирована и так и останется

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.5,
            max_tokens=150,
            timeout=60.0 # Таймаут в секундах
        )
        
        # log_info(f"Ответ от сервера OpenAI получен.") 
        # log_debug(f"Объект ответа (сырой):\n------ ОБЪЕКТ ОТВЕТА НАЧАЛО ------\n{response}\n------ ОБЪЕКТ ОТВЕТА КОНЕЦ ------")

        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            if message and message.content:
                response_text = message.content.strip()
                log_info(f"Извлеченный текст ответа: {response_text}")
                return response_text
            else:
                log_warning("Ответ от ChatGPT не содержит 'message.content'.")
                return "Ответ от ChatGPT не содержит ожидаемого текста."
        else:
            log_warning("Ответ от ChatGPT не содержит 'choices'.")
            return "Не удалось получить корректный ответ от ChatGPT (нет 'choices')."

    except openai.APIConnectionError as e:
        log_error(f"Ошибка соединения с API OpenAI: {e}", e)
        return f"Ошибка соединения с OpenAI: {e}"
    except openai.RateLimitError as e:
        log_error(f"Превышен лимит запросов к API OpenAI: {e}", e)
        return f"Превышен лимит запросов к OpenAI: {e}"
    except openai.AuthenticationError as e:
        log_error(f"Ошибка аутентификации с API OpenAI (проверьте API-ключ): {e}", e)
        error_details = str(e)
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'): # Проверяем, есть ли json метод
            try:
                error_json = e.response.json()
                if 'error' in error_json and 'message' in error_json['error']:
                    error_details = error_json['error']['message']
            except Exception: # Не удалось распарсить json или другая ошибка
                pass # Используем error_details, который уже str(e)
        return f"Ошибка аутентификации с OpenAI: {error_details}"
    except openai.APIStatusError as e: # Обработка других ошибок API (например, 4xx, 5xx, кроме 401, 429)
        log_error(f"Ошибка API OpenAI: статус {e.status_code}, ответ: {e.response}", e)
        error_message = f"Ошибка API OpenAI (статус {e.status_code})"
        if hasattr(e, 'response') and e.response:
            try:
                # Попытка получить JSON из ответа, если возможно
                if hasattr(e.response, 'json'): 
                    error_json = e.response.json()
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message += f": {error_json['error']['message']}"
                    else:
                        # Если нет стандартной структуры, пытаемся получить текст ответа
                        error_message += f": {e.response.text}"
                else:
                     error_message += f": {e.response.text}" # Если response не имеет метода json()
            except Exception:
                 error_message += f": {e.response.text if hasattr(e.response, 'text') else 'Нет текста ошибки'}" # Если не json или другая ошибка при парсинге
        return error_message
    except openai.APITimeoutError as e: # Явный таймаут от библиотеки openai
        log_error(f"Таймаут запроса к API OpenAI (openai library): {e}", e)
        return f"Таймаут при обращении к OpenAI (openai library): {e}"
    except Exception as e: # Общая непредвиденная ошибка
        log_error(f"Непредвиденная ошибка при вызове API ChatGPT через библиотеку openai: {type(e).__name__} - {e}", e)
        return f"Непредвиденная ошибка (openai library): {type(e).__name__} - {str(e)}"

class RSLRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознаватель русского жестового языка")
        self.setGeometry(100, 100, 1000, 700)
        
        log_info("Приложение запущено")
        
        # Основные компоненты
        self.capture = None
        self.recognizer = None
        self.frame_counter = 0
        self.tensors_list = [] # Добавил обратно, т.к. используется в update_frame для сбора кадров для модели
        
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
        # self.text_display.setText("Жду ввода предложения...") # Будет установлено в _init_ui или _reset_state
        
    def _init_ui(self):
        # Создание центрального виджета
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Основной вертикальный компоновщик
        main_layout = QVBoxLayout(central_widget)

        # --- Панель выбора модели и конфигурации ---
        settings_layout = QHBoxLayout()

        # Выбор модели
        model_group_layout = QVBoxLayout()
        model_label = QLabel("Модель:", self)
        model_group_layout.addWidget(model_label)
        self.model_selector = QComboBox(self)
        if self.available_models:
            self.model_selector.addItems(self.available_models)
        else:
            self.model_selector.addItem("Модели не найдены")
        model_group_layout.addWidget(self.model_selector)
        self.model_path_button = QPushButton("Обзор (модель)...", self)
        self.model_path_button.clicked.connect(self.browse_model)
        model_group_layout.addWidget(self.model_path_button)
        settings_layout.addLayout(model_group_layout)

        # Выбор конфигурации
        config_group_layout = QVBoxLayout()
        config_label = QLabel("Конфигурация:", self)
        config_group_layout.addWidget(config_label)
        self.config_selector = QComboBox(self)
        if self.available_configs:
            self.config_selector.addItems(self.available_configs)
        else:
            self.config_selector.addItem("Конфигурации не найдены")
        config_group_layout.addWidget(self.config_selector)
        self.config_path_button = QPushButton("Обзор (конфиг)...", self)
        self.config_path_button.clicked.connect(self.browse_config)
        config_group_layout.addWidget(self.config_path_button)
        settings_layout.addLayout(config_group_layout)
        
        main_layout.addLayout(settings_layout)
        
        # Виджет для отображения видео
        self.video_widget = QLabel(self)
        self.video_widget.setMinimumSize(640, 480)
        self.video_widget.setAlignment(Qt.AlignCenter)
        self.video_widget.setText("Видео не запущено")
        main_layout.addWidget(self.video_widget)
        
        # Текстовое поле для отображения результатов распознавания
        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(100)
        self.text_display.setText("Жду ввода предложения...")
        main_layout.addWidget(self.text_display)
        
        # Кнопки управления
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Запустить распознавание", self)
        self.start_button.clicked.connect(self.start_recognition)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Остановить распознавание", self)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        self.record_button = QPushButton("Записать видео", self)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        controls_layout.addWidget(self.record_button)
        
        main_layout.addLayout(controls_layout)

        # Дополнительные кнопки (Сохранить, Сбросить)
        extra_controls_layout = QHBoxLayout()
        self.save_button = QPushButton("Сохранить результат", self)
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        extra_controls_layout.addWidget(self.save_button)
        
        self.reset_button = QPushButton("Сбросить", self)
        self.reset_button.clicked.connect(self._reset_state)
        extra_controls_layout.addWidget(self.reset_button)

        main_layout.addLayout(extra_controls_layout)

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
            # Восстанавливаем кнопки, чтобы пользователь мог выбрать другую модель/конфиг
            self._reset_state_before_start() # Сброс переменных
            self.start_button.setEnabled(True)
            self.model_selector.setEnabled(True)
            self.config_selector.setEnabled(True)
            self.model_path_button.setEnabled(True)
            self.config_path_button.setEnabled(True)
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
        self.timer.start(30)
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
            f"На каждом из {len(self.current_sentence_predictions)} шагов распознавания тебе будет предоставлен список из нескольких (до {self.recognizer.num_top_predictions}) наиболее вероятных жестов и их вероятностей.",
            "Твоя задача — проанализировать всю последовательность, учитывая как уверенность отдельных жестов на каждом шаге, так и общий контекст, чтобы сформировать наиболее вероятное и грамматически корректное предложение на русском языке.",
            "Выбирай наиболее подходящие жесты из предложенных на каждом шаге, чтобы составить связное предложение.",
            "Даже если некоторые жесты на каком-то шаге имеют низкую уверенность или кажутся выбивающимися из контекста, постарайся их интерпретировать в рамках возможного предложения, возможно, выбрав другой вариант с этого же шага или учитывая жесты с предыдущих/последующих шагов.",
            "Отдай приоритет жестам с более высокой уверенностью, но не игнорируй контекст, который могут создавать другие жесты.",
            "Избегай повторения одного и того же жеста подряд, если это не выглядит осмысленно в контексте предложения.",
            "Постарайся составить лаконичное, но полное предложение.",
            f"Входные данные (последовательность шагов, каждый шаг - список из до {self.recognizer.num_top_predictions} кортежей [жест, вероятность]):"
        ]
        
        # current_sentence_predictions теперь это список списков: [ [('ж1',в1), ('ж2',в2)], [('ж1',в1),('ж2',в2)] ]
        for i, step_top_n_predictions in enumerate(self.current_sentence_predictions):
            # step_top_n_predictions это, например, [('жест1', 0.7), ('жест2', 0.15), ('жест3', 0.05)]
            predictions_str_list = []
            if step_top_n_predictions: # Убедимся, что список не пуст
                for gloss, confidence in step_top_n_predictions:
                    predictions_str_list.append(f"('{gloss}', {confidence:.3f})")
                step_details = f"[{', '.join(predictions_str_list)}]"
            else:
                step_details = "[Нет данных на шаге]"
            prompt_lines.append(f"Шаг {i+1}: {step_details}")
            
        prompt_lines.append("Пожалуйста, составь одно законченное предложение на русском языке на основе этих данных.")
        chatgpt_prompt = "\n".join(prompt_lines)
        
        log_info("Сгенерированный промпт для языковой модели:")
        # Для краткости лога, можно не выводить весь промпт каждый раз или выводить его часть
        # log_info(chatgpt_prompt) 
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [PROMPT] Сформирован промпт из {len(self.current_sentence_predictions)} шагов.")

        # Получаем ответ от ChatGPT
        final_sentence = get_chatgpt_response(chatgpt_prompt)
            
        log_info(f"Итоговое предложение от ChatGPT: {final_sentence}")
        
        self.last_processed_data_for_saving = {
            'prompt': chatgpt_prompt,
            'raw_predictions': list(self.current_sentence_predictions), 
            'final_sentence': final_sentence
        }
        
        self.text_display.setText(f"Результат: {final_sentence}") # Убрано упоминание промпта, т.к. теперь это реальный результат
        
        if final_sentence and not final_sentence.startswith("Ошибка при обращении к ChatGPT"):
            self.save_button.setEnabled(True)
            
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
        
        # Добавляем на кадр распознанный жест
        frame_with_text = frame.copy()
        
        # Используем self.last_recognized_gloss_for_overlay и self.last_recognized_confidence_for_overlay
        # для наложения текста на видеокадр, а не старые prediction_list/confidence_list
        current_gloss_to_display = self.last_recognized_gloss_for_overlay
        current_confidence_to_display = self.last_recognized_confidence_for_overlay

        if current_gloss_to_display != self.NO_GESTURE_SIGNAL: # Проверяем, что есть что отображать
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
                    
                    # Получаем результат распознавания - теперь это список топ-N предсказаний
                    # например, [('жест1', 0.7), ('жест2', 0.15), ('жест3', 0.05)]
                    top_n_predictions = self.recognizer.predict(input_tensor)

                    if not top_n_predictions: # Если predict вернул пустой список (ошибка или нет предсказаний)
                        log_warning("Recognizer.predict() не вернул предсказаний.")
                        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
                        self.last_recognized_confidence_for_overlay = 0.0
                        # Можно добавить обработку такой ситуации, если необходимо
                    else:
                        # Берем топ-1 предсказание для оверлея на видео и первичной логики
                        gloss, confidence = top_n_predictions[0]

                        # Обновляем информацию для оверлея на видео
                        self.last_recognized_gloss_for_overlay = gloss
                        self.last_recognized_confidence_for_overlay = confidence
                        
                        log_prediction(gloss, confidence, input_shape=input_tensor.shape) # Логируем топ-1
                        if self.recognizer.debug_mode and len(top_n_predictions) > 1:
                            # Дополнительно логируем остальные из топ-N, если они есть
                            log_debug(f"    Топ-{len(top_n_predictions)} предсказания на шаге (из main.py):")
                            for i, (g, c) in enumerate(top_n_predictions):
                                log_debug(f"      {i+1}. {g}: {c:.4f}")
                        
                        if gloss == self.NO_GESTURE_SIGNAL:
                            if self.is_collecting_sentence:
                                log_info("Обнаружен сигнал NO_GESTURE ('---'), завершение сбора предложения.")
                                self._process_collected_sentence() 
                        else: # Распознан содержательный жест (топ-1 не "---")
                            # Проверяем уверенность топ-1 жеста по порогу
                            if confidence < self.recognizer.confidence_threshold: 
                                log_debug(f"Топ-1 жест '{gloss}' ({confidence:.2f}) отброшен из-за низкой уверенности ( < {self.recognizer.confidence_threshold}). Весь набор топ-{len(top_n_predictions)} для этого шага игнорируется.")
                            else:
                                # Уверенность топ-1 жеста достаточная, обрабатываем весь набор топ-N
                                if not self.is_collecting_sentence:
                                    log_info(f"Обнаружен первый содержательный топ-1 жест '{gloss}' ({confidence:.2f}), начало сбора предложения.")
                                    self.is_collecting_sentence = True
                                    self.current_sentence_predictions = [] 
                                    self.text_display.setText("Идет набор предложения...")
                                    self.statusBar().showMessage("Идет набор предложения...")
                                    self.save_button.setEnabled(False) 

                                if self.is_collecting_sentence:
                                    # Добавляем ВЕСЬ СПИСОК топ-N предсказаний для этого шага
                                    self.current_sentence_predictions.append(list(top_n_predictions)) 
                                    log_debug(f"Набор из топ-{len(top_n_predictions)} предсказаний (начиная с '{gloss}' ({confidence:.2f})) добавлен в текущее предложение (шаг {len(self.current_sentence_predictions)}).")
                    
                    self.tensors_list = [] 
                    
                except Exception as e:
                    self.tensors_list = []  # Очищаем буфер в случае ошибки
                    log_error("Ошибка при обработке кадров для предсказания", e)
                    self.statusBar().showMessage(f"Ошибка обработки предсказания: {e}")
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
        
        # Используем self.last_recognized_gloss_for_overlay и self.last_recognized_confidence_for_overlay
        # для наложения текста на видеокадр, а не старые prediction_list/confidence_list
        current_gloss_to_display = self.last_recognized_gloss_for_overlay
        current_confidence_to_display = self.last_recognized_confidence_for_overlay

        if current_gloss_to_display != self.NO_GESTURE_SIGNAL: # Проверяем, что есть что отображать
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
                    
                    # Получаем результат распознавания - теперь это список топ-N предсказаний
                    # например, [('жест1', 0.7), ('жест2', 0.15), ('жест3', 0.05)]
                    top_n_predictions = self.recognizer.predict(input_tensor)

                    if not top_n_predictions: # Если predict вернул пустой список (ошибка или нет предсказаний)
                        log_warning("Recognizer.predict() не вернул предсказаний.")
                        self.last_recognized_gloss_for_overlay = self.NO_GESTURE_SIGNAL
                        self.last_recognized_confidence_for_overlay = 0.0
                        # Можно добавить обработку такой ситуации, если необходимо
                    else:
                        # Берем топ-1 предсказание для оверлея на видео и первичной логики
                        gloss, confidence = top_n_predictions[0]

                        # Обновляем информацию для оверлея на видео
                        self.last_recognized_gloss_for_overlay = gloss
                        self.last_recognized_confidence_for_overlay = confidence
                        
                        log_prediction(gloss, confidence, input_shape=input_tensor.shape) # Логируем топ-1
                        if self.recognizer.debug_mode and len(top_n_predictions) > 1:
                            # Дополнительно логируем остальные из топ-N, если они есть
                            log_debug(f"    Топ-{len(top_n_predictions)} предсказания на шаге (из main.py):")
                            for i, (g, c) in enumerate(top_n_predictions):
                                log_debug(f"      {i+1}. {g}: {c:.4f}")
                        
                        if gloss == self.NO_GESTURE_SIGNAL:
                            if self.is_collecting_sentence:
                                log_info("Обнаружен сигнал NO_GESTURE ('---'), завершение сбора предложения.")
                                self._process_collected_sentence() 
                        else: # Распознан содержательный жест (топ-1 не "---")
                            # Проверяем уверенность топ-1 жеста по порогу
                            if confidence < self.recognizer.confidence_threshold: 
                                log_debug(f"Топ-1 жест '{gloss}' ({confidence:.2f}) отброшен из-за низкой уверенности ( < {self.recognizer.confidence_threshold}). Весь набор топ-{len(top_n_predictions)} для этого шага игнорируется.")
                            else:
                                # Уверенность топ-1 жеста достаточная, обрабатываем весь набор топ-N
                                if not self.is_collecting_sentence:
                                    log_info(f"Обнаружен первый содержательный топ-1 жест '{gloss}' ({confidence:.2f}), начало сбора предложения.")
                                    self.is_collecting_sentence = True
                                    self.current_sentence_predictions = [] 
                                    self.text_display.setText("Идет набор предложения...")
                                    self.statusBar().showMessage("Идет набор предложения...")
                                    self.save_button.setEnabled(False) 

                                if self.is_collecting_sentence:
                                    # Добавляем ВЕСЬ СПИСОК топ-N предсказаний для этого шага
                                    self.current_sentence_predictions.append(list(top_n_predictions)) 
                                    log_debug(f"Набор из топ-{len(top_n_predictions)} предсказаний (начиная с '{gloss}' ({confidence:.2f})) добавлен в текущее предложение (шаг {len(self.current_sentence_predictions)}).")
                    
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