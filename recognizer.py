#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для распознавания русского жестового языка
"""

import os
import json
import cv2
import numpy as np
import onnxruntime as ort

class RSLRecognizer:
    """Класс для распознавания русского жестового языка"""
    
    def __init__(self, model_path=None, config=None, classes_path=None):
        """
        Инициализирует распознаватель РЖЯ
        
        Args:
            model_path (str): Путь к ONNX модели
            config (dict): Конфигурационные параметры
            classes_path (str): Путь к JSON файлу с классами
        """
        self.model_path = model_path
        self.config = config or {}
        self.session = None
        self.input_name = None
        self.input_shape = None 
        self.window_size = None
        self.output_names = None
        self.classes = {}
        
        # Параметры из конфигурации
        self.frame_interval = self.config.get('frame_interval', 2)
        self.mean = self.config.get('mean', [123.675, 116.28, 103.53])
        self.std = self.config.get('std', [58.395, 57.12, 57.375])
        
        # Новые параметры из конфигурации
        self.confidence_threshold = self.config.get('confidence_threshold', 0.2)
        self.smoothing_window = self.config.get('smoothing_window', 3)
        self.debug_mode = self.config.get('debug_mode', False)
        self.expected_output_shape = self.config.get('output_shape', None)
        
        # История предсказаний для сглаживания
        self.prediction_history = []
        
        # Загрузка словаря классов
        if classes_path and os.path.exists(classes_path):
            self.load_classes(classes_path)
        
        # Создание пустого словаря классов, если файл не существует
        if not os.path.exists("classes.json"):
            self.create_empty_classes_file()
        
        # Загрузка модели
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_classes(self, classes_path):
        """
        Загружает словарь классов из JSON файла
        
        Args:
            classes_path (str): Путь к JSON файлу с классами
        
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                # Загружаем данные из JSON
                classes_data = json.load(f)
                
                # Проверяем, что содержимое - словарь
                if not isinstance(classes_data, dict):
                    print(f"Ошибка: Файл {classes_path} не содержит словарь")
                    return False
                
                # Очищаем текущий словарь классов
                self.classes = {}
                
                # Преобразуем ключи из строк в числа и фильтруем некорректные записи
                for key, value in classes_data.items():
                    try:
                        # Попытка преобразовать ключ в целое число
                        idx = int(key)
                        # Проверка, что значение - строка
                        if isinstance(value, str):
                            self.classes[idx] = value
                    except (ValueError, TypeError) as e:
                        print(f"Пропущен некорректный ключ '{key}': {e}")
                
                print(f"Загружено {len(self.classes)} классов из {classes_path}")
                return True
                
        except json.JSONDecodeError as e:
            print(f"Ошибка при разборе JSON файла {classes_path}: {e}")
            # Создаем пустой файл, если существующий поврежден
            self.create_default_classes_file(classes_path)
            return False
        except FileNotFoundError:
            print(f"Файл классов {classes_path} не найден")
            # Создаем новый файл, так как указанный не существует
            self.create_default_classes_file(classes_path)
            return False
        except Exception as e:
            print(f"Ошибка загрузки словаря классов: {e}")
            return False
    
    def create_empty_classes_file(self, output_file="classes.json"):
        """
        Создает пустой файл classes.json, если он отсутствует
        
        Args:
            output_file (str): Путь к файлу classes.json
        """
        return self.create_default_classes_file(output_file, empty=True)
    
    def create_default_classes_file(self, output_file="classes.json", empty=False):
        """
        Создает файл classes.json с базовыми классами или пустой словарь
        
        Args:
            output_file (str): Путь к файлу classes.json
            empty (bool): Если True, создает пустой файл, иначе заполняет базовыми классами
        
        Returns:
            bool: True, если файл создан успешно, иначе False
        """
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            if empty:
                # Создаем пустой словарь
                default_classes = {}
                message = "пустой файл"
            else:
                # Создаем словарь с базовыми классами
                default_classes = {
                    0: "Нет жеста",
                    1: "Жест 1",
                    2: "Жест 2",
                    3: "Жест 3",
                    4: "Жест 4",
                    5: "Жест 5",
                    # Можно добавить другие базовые классы при необходимости
                }
                message = "файл с базовыми классами"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(default_classes, f, ensure_ascii=False, indent=4)
            
            # Обновляем словарь классов
            self.classes = default_classes
            
            print(f"Создан {message} {output_file}")
            return True
        except Exception as e:
            print(f"Ошибка при создании файла классов: {e}")
            return False
    
    def load_model(self, model_path):
        """
        Загружает ONNX модель для распознавания жестов с поддержкой GPU
        
        Args:
            model_path (str): Путь к файлу модели
        
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        try:
            # Отображаем доступные провайдеры
            available_providers = ort.get_available_providers()
            print(f"Доступные провайдеры ONNX Runtime: {available_providers}")
            
            # Создаем сессию с указанным провайдером в зависимости от доступности
            if 'CUDAExecutionProvider' in available_providers:
                # Если CUDA доступен, используем его с опциями
                providers = ['CUDAExecutionProvider']
                provider_options = [{'device_id': 0}]  # Опции для CUDA провайдера
                
                # Создаем сессию с указанными провайдерами и опциями
                try:
                    self.session = ort.InferenceSession(
                        model_path, 
                        providers=providers,
                        provider_options=provider_options
                    )
                    print(f"Модель {os.path.basename(model_path)} загружена на GPU")
                except Exception as e:
                    print(f"Ошибка при загрузке на GPU: {e}")
                    print("Пробуем загрузить модель на CPU...")
                    self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    print(f"Модель {os.path.basename(model_path)} загружена на CPU")
            else:
                # Если CUDA недоступен, используем только CPU
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print(f"GPU не доступен, модель {os.path.basename(model_path)} загружена на CPU")
            
            # Получаем информацию о входном и выходном тензорах
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[3]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Выводим используемые провайдеры
            used_providers = self.session.get_providers()
            print(f"Фактически используемые провайдеры: {used_providers}")
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def predict(self, frames_tensor):
        """
        Распознаёт жест по последовательности кадров
        
        Args:
            frames_tensor (numpy.ndarray): Тензор с последовательностью кадров
        
        Returns:
            tuple: (метка, уверенность)
        """
        if self.session is None:
            return None, 0
        
        try:
            # Получаем выходные данные модели
            outputs = self.session.run(self.output_names, {self.input_name: frames_tensor.astype(np.float32)})
            
            # Получаем первый выходной тензор
            output = outputs[0]
            
            # Проверяем, что вывод не пустой
            if output is None or output.size == 0:
                print("Предупреждение: Пустой выходной тензор от модели")
                return None, 0
                
            # Вывод отладочной информации
            if self.debug_mode:
                print(f"Форма выходного тензора: {output.shape}")
            
            # Если задана ожидаемая форма выходного тензора, проверяем соответствие
            if self.expected_output_shape is not None:
                expected_shape = self.expected_output_shape
                if len(expected_shape) == 2:
                    batch_size, num_classes = expected_shape
                    
                    # Если выходной тензор не соответствует ожидаемой форме, пробуем преобразовать
                    if len(output.shape) != 2 or output.shape[1] != num_classes:
                        if self.debug_mode:
                            print(f"Форма тензора {output.shape} не соответствует ожидаемой {expected_shape}")
                        
                        # Если это одномерный тензор, преобразуем в двумерный
                        if len(output.shape) == 1 and output.shape[0] == num_classes:
                            output = output.reshape(1, -1)
                        # Если размер меньше ожидаемого, используем его как есть
                        elif output.size < num_classes:
                            pass
                        # Если размер больше ожидаемого, обрезаем
                        else:
                            output = output[:batch_size, :num_classes]
            
            # Проверяем форму выходного тензора
            # Если вывод - одномерный массив, используем его напрямую
            if len(output.shape) == 1:
                # Используем напрямую одномерный массив вероятностей
                logits = output
            # Если вывод 2D (batch, classes), берем первый элемент батча
            elif len(output.shape) == 2:
                logits = output[0]
            # Для других форм пытаемся привести к плоскому массиву
            else:
                # Пытаемся сжать тензор до одномерного
                logits = np.ravel(output)
                if self.debug_mode:
                    print(f"Преобразован тензор размерности {output.shape} в плоский массив {logits.shape}")
            
            # Проверяем, что у нас есть какие-то данные
            if logits.size == 0:
                print("Предупреждение: Пустой массив логитов после обработки")
                return None, 0
                
            # Если значений слишком мало (меньше чем классов), возможно что-то не так с выводом
            if logits.size < 10:  # Предполагаем, что должно быть хотя бы 10 классов
                if self.debug_mode:
                    print(f"Предупреждение: Подозрительно малый размер вывода: {logits.size}")
                # В этом случае можем вернуть заглушку
                return "Недостаточно данных", 0.1
            
            # Находим индекс максимального значения
            try:
                max_idx = int(np.argmax(logits))
                # Проверяем, что индекс не выходит за границы массива
                if max_idx >= logits.size:
                    print(f"Ошибка: Индекс {max_idx} выходит за границы массива размера {logits.size}")
                    return None, 0
                
                confidence = float(logits[max_idx])
                
                # Нормализуем значение уверенности с помощью softmax, если значение слишком большое
                if confidence > 10:  # Вероятно, это логиты, а не вероятности
                    # Применяем softmax для получения вероятностей
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    max_idx = int(np.argmax(probs))
                    confidence = float(probs[max_idx])
                
                # Проверяем порог уверенности
                if confidence < self.confidence_threshold:
                    return None, confidence
                
                # Извлекаем метку из словаря классов
                if self.classes and max_idx in self.classes:
                    label = self.classes[max_idx]
                else:
                    # Просто возвращаем номер класса, если метки нет в словаре
                    label = f"Класс {max_idx}"
                
                # Применяем сглаживание предсказаний, если включено
                if self.smoothing_window > 1:
                    # Добавляем текущее предсказание в историю
                    self.prediction_history.append((label, confidence))
                    # Ограничиваем историю размером окна сглаживания
                    if len(self.prediction_history) > self.smoothing_window:
                        self.prediction_history = self.prediction_history[-self.smoothing_window:]
                    
                    # Если история достаточно заполнена, применяем сглаживание
                    if len(self.prediction_history) >= self.smoothing_window:
                        # Подсчитываем частоту каждой метки в окне
                        label_counts = {}
                        for hist_label, hist_conf in self.prediction_history:
                            label_counts[hist_label] = label_counts.get(hist_label, 0) + 1
                        
                        # Находим наиболее частую метку
                        most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                        
                        # Если текущая метка не совпадает с наиболее частой, 
                        # и наиболее частая встречается достаточно часто, используем её
                        if label != most_common_label and label_counts[most_common_label] >= self.smoothing_window // 2:
                            if self.debug_mode:
                                print(f"Сглаживание: заменена метка {label} на {most_common_label}")
                            label = most_common_label
                
                return label, confidence
                
            except IndexError as e:
                print(f"Ошибка индексации в массиве логитов: {e}")
                if self.debug_mode:
                    print(f"Размер массива: {logits.size}, форма: {logits.shape}")
                return None, 0
                
        except IndexError as e:
            # Конкретная обработка ошибок индексации
            print(f"Ошибка индексации при предсказании: {e}")
            return None, 0
        except Exception as e:
            # Общая обработка других ошибок
            print(f"Ошибка при предсказании: {e}")
            return None, 0
    
    @staticmethod
    def resize(im, new_shape=(224, 224)):
        """
        Изменяет размер изображения с сохранением пропорций и добавлением отступов
        
        Args:
            im (numpy.ndarray): Исходное изображение
            new_shape (tuple): Новый размер (высота, ширина)
        
        Returns:
            numpy.ndarray: Измененное изображение
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        return im


# Тестирование модуля при прямом запуске
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование модуля распознавания РЖЯ")
    parser.add_argument("--model", required=True, help="Путь к ONNX модели")
    parser.add_argument("--classes", default="classes.json", help="Путь к JSON файлу с классами")
    parser.add_argument("--config", help="Путь к файлу конфигурации YAML")
    
    args = parser.parse_args()
    
    # Загрузка конфигурации, если указана
    config = None
    if args.config:
        import yaml
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
    
    # Создание распознавателя
    recognizer = RSLRecognizer(args.model, config, args.classes)
    
    print(f"Модель загружена: {recognizer.session is not None}")
    print(f"Размер окна: {recognizer.window_size}")
    print(f"Количество классов: {len(recognizer.classes)}")
    print(f"Интервал кадров: {recognizer.frame_interval}")
    
    # Простой тест на случайных данных
    if recognizer.session:
        random_frames = np.random.rand(1, 1, 3, recognizer.window_size, 224, 224).astype(np.float32)
        label, confidence = recognizer.predict(random_frames)
        print(f"Тестовое предсказание: {label} (уверенность: {confidence:.4f})") 