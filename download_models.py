#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для скачивания моделей
"""

import os
import sys
import time
import urllib.request
import argparse
from pathlib import Path

# Отключаем буферизацию вывода
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Глобальные параметры
FORCE_DOWNLOAD = False  # Принудительно перезагружать файлы
AUTO_YES = False  # Автоматически отвечать "да" на вопросы

# Настройки терминала
def setup_terminal():
    """Настраивает терминал для корректного отображения прогресс-бара"""
    # Проверяем, является ли вывод терминалом
    if sys.stdout.isatty():
        # Отключаем автоматический перенос строк в терминале
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'raw'):
            try:
                sys.stdout.buffer.raw.write(b'\033[?7l')
                return True
            except:
                pass
    return False

def restore_terminal():
    """Восстанавливает настройки терминала"""
    if sys.stdout.isatty():
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'raw'):
            try:
                sys.stdout.buffer.raw.write(b'\033[?7h')
            except:
                pass

# URLs моделей
MODEL_URLS = {
    "mvit16-4.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit16-4.onnx",
    "mvit32-2.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx",
    "mvit48-2.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit48-2.onnx",
    "swin16-3.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin16-3.onnx",
    "swin32-2.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin32-2.onnx",
    "swin48-1.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin48-1.onnx", 
    "resnet16-3.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet16-3.onnx",
    "resnet32-2.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet32-2.onnx",
    "resnet48-1.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet48-1.onnx",
    "SignFlow-A.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/SignFlow-A.onnx",
    "SignFlow-R.onnx": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/SignFlow-R.onnx",
}

# Размеры файлов в байтах (приблизительно)
MODEL_SIZES = {
    "mvit16-4.onnx": 140_000_000,  # ~140 MB
    "mvit32-2.onnx": 140_000_000,  # ~140 MB
    "mvit48-2.onnx": 141_000_000,  # ~141 MB
    "swin16-3.onnx": 821_000_000,  # ~821 MB
    "swin32-2.onnx": 821_000_000,  # ~821 MB
    "swin48-1.onnx": 821_000_000,  # ~821 MB
    "resnet16-3.onnx": 146_000_000,  # ~146 MB
    "resnet32-2.onnx": 146_000_000,  # ~146 MB
    "resnet48-1.onnx": 146_000_000,  # ~146 MB
    "SignFlow-A.onnx": 140_000_000,  # ~140 MB (приблизительно)
    "SignFlow-R.onnx": 140_000_000,  # ~140 MB (приблизительно)
}

class DownloadProgressBar:
    def __init__(self):
        self.last_time = None
        self.last_count = None
        self.last_percent = -1  # Последний отображенный процент
        self.dots = 0
        # Важно: не запрашиваем размер терминала, если вывод перенаправлен
        self.bar_length = 30
    
    def __call__(self, count, block_size, total_size):
        current_size = count * block_size
        
        # Проверяем, нужно ли обновлять прогресс
        if total_size > 0:
            current_percent = min(int(current_size * 100 / total_size), 100)
            # Обновляем прогресс только если процент изменился или прошло достаточно времени
            if current_percent == self.last_percent:
                # Если процент не изменился, проверяем, прошло ли достаточно времени
                if self.last_time and time.time() - self.last_time < 0.5:
                    return  # Пропускаем обновление, если с момента последнего прошло менее 0.5 секунды
            self.last_percent = current_percent
        elif self.last_time and time.time() - self.last_time < 0.5:
            return  # Для неизвестного размера также ограничиваем частоту обновлений
        
        # Вычисляем скорость скачивания
        speed = "N/A"
        if self.last_time and self.last_count:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed > 0:
                bytes_per_sec = (count - self.last_count) * block_size / elapsed
                if bytes_per_sec > 1024*1024:
                    speed = f"{bytes_per_sec / (1024*1024):.1f} MB/s"
                elif bytes_per_sec > 1024:
                    speed = f"{bytes_per_sec / 1024:.1f} KB/s"
                else:
                    speed = f"{bytes_per_sec:.0f} B/s"
        
        # Запоминаем текущее время и счетчик
        self.last_time = time.time()
        self.last_count = count
        
        # Отображаем прогресс
        mb_current = current_size / (1024 * 1024)
        
        if total_size > 0:
            mb_total = total_size / (1024 * 1024)
            percent = min(int(current_size * 100 / total_size), 100)
            filled_length = int(self.bar_length * percent / 100)
            bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
            message = f"\r|{bar}| {percent:3d}% [{mb_current:.1f}/{mb_total:.1f} MB] {speed}"
        else:
            # Если размер не известен, показываем анимацию
            self.dots = (self.dots + 1) % 4
            dots_str = '.' * self.dots + ' ' * (3 - self.dots)
            message = f"\rЗагрузка{dots_str} {mb_current:.1f} MB {speed}"
        
        # Выводим сообщение и сбрасываем буфер
        sys.stdout.write(message)
        sys.stdout.flush()

def download_model(model_name, output_dir, max_retries=3):
    """Скачивает модель по указанному URL"""
    if model_name not in MODEL_URLS:
        print(f"Ошибка: Модель {model_name} не найдена в списке доступных моделей")
        return False
    
    url = MODEL_URLS[model_name]
    output_path = os.path.join(output_dir, model_name)
    
    # Если включен режим принудительной загрузки и файл существует - удаляем его
    if FORCE_DOWNLOAD and os.path.exists(output_path):
        print(f"Режим принудительной загрузки: удаляем существующий файл {model_name}")
        os.remove(output_path)
    
    # Проверяем наличие файла и его размер
    expected_size = MODEL_SIZES.get(model_name, 0)
    
    if os.path.exists(output_path) and not FORCE_DOWNLOAD:
        file_size = os.path.getsize(output_path)
        
        # Проверяем размер файла
        if expected_size > 0:
            # Если файл существует и размер соответствует ожидаемому (в пределах 1 МБ)
            if abs(file_size - expected_size) < 1024*1024:
                print(f"Модель {model_name} уже загружена ({file_size/(1024*1024):.1f} МБ) - пропускаем")
                return True
            # Если файл существует, но размер немного меньше - возможно, загрузка была прервана
            elif file_size < expected_size and file_size > expected_size * 0.95:
                print(f"Найден частично загруженный файл ({file_size/(1024*1024):.1f} МБ из {expected_size/(1024*1024):.1f} МБ)")
                if AUTO_YES:
                    print("Автоматически продолжаем загрузку (--yes)")
                    os.remove(output_path)
                else:
                    user_response = input("Хотите продолжить загрузку? (y/n): ").strip().lower()
                    if user_response != 'y':
                        print("Пропускаем загрузку модели")
                        return True
                    print("Удаляем неполный файл и начинаем загрузку заново...")
                    os.remove(output_path)
        else:
            # Если ожидаемый размер неизвестен, но файл существует
            print(f"Файл {model_name} уже существует (размер: {file_size/(1024*1024):.1f} МБ)")
            if AUTO_YES:
                print("Автоматически пропускаем загрузку (--yes)")
                return True
            user_response = input("Хотите загрузить заново? (y/n): ").strip().lower()
            if user_response != 'y':
                print("Пропускаем загрузку модели")
                return True
            print("Удаляем существующий файл и начинаем загрузку заново...")
            os.remove(output_path)
    
    print(f"Скачивание модели {model_name}...")
    
    for retry in range(max_retries):
        try:
            if retry > 0:
                print(f"\nПопытка {retry+1} из {max_retries}...")
            
            # Создаем новый объект для отслеживания прогресса
            progress_bar = DownloadProgressBar()
            
            # Настраиваем таймаут для соединения
            opener = urllib.request.build_opener()
            opener.addheaders = [
                ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
            ]
            urllib.request.install_opener(opener)
            
            # Скачиваем файл с прогресс-баром и таймаутом
            urllib.request.urlretrieve(url, output_path, reporthook=progress_bar)
            
            # Новая строка после прогресс-бара
            sys.stdout.write('\n')
            sys.stdout.flush()
            
            # Проверяем, что файл загружен корректно
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                print(f"Модель {model_name} успешно скачана ({file_size/(1024*1024):.1f} МБ)")
                return True
            else:
                print(f"Ошибка: Файл пустой или не существует")
                continue  # Пробуем снова если есть попытки
            
        except urllib.error.URLError as e:
            print(f"\nОшибка сети при скачивании модели {model_name}: {e}")
            if hasattr(e, 'reason'):
                print(f"Причина: {e.reason}")
            
            if retry < max_retries - 1:  # Если это не последняя попытка
                delay = 2 ** retry  # Экспоненциальная задержка: 1, 2, 4 секунды и т.д.
                print(f"Повторная попытка через {delay} сек...")
                time.sleep(delay)
            else:
                print(f"Исчерпаны все попытки скачивания модели {model_name}")
                return False
                
        except Exception as e:
            print(f"\nОшибка при скачивании модели {model_name}: {e}")
            if retry < max_retries - 1:  # Если это не последняя попытка
                delay = 2 ** retry
                print(f"Повторная попытка через {delay} сек...")
                time.sleep(delay)
            else:
                print(f"Исчерпаны все попытки скачивания модели {model_name}")
                return False
    
    return False  # Если все попытки закончились неудачей

def main():
    # Настраиваем терминал
    terminal_configured = setup_terminal()
    
    try:
        parser = argparse.ArgumentParser(description="Скачивание моделей")
        parser.add_argument("--model", choices=list(MODEL_URLS.keys()) + ["all"], default="all",
                            help="Имя модели для скачивания (по умолчанию: all)")
        parser.add_argument("--output", default="models", help="Директория для сохранения моделей (по умолчанию: models)")
        parser.add_argument("--force", action="store_true", help="Принудительно перезагрузить модели, даже если они уже существуют")
        parser.add_argument("--yes", "-y", action="store_true", help="Автоматически отвечать 'да' на все вопросы")
        
        args = parser.parse_args()
        
        # Создаем директорию для моделей, если она не существует
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Устанавливаем глобальные переменные для неинтерактивного режима
        global FORCE_DOWNLOAD, AUTO_YES
        FORCE_DOWNLOAD = args.force
        AUTO_YES = args.yes
        
        if args.model == "all":
            print(f"Скачивание всех доступных моделей в директорию {output_dir}...")
            success_count = 0
            total_models = len(MODEL_URLS)
            
            for i, model_name in enumerate(MODEL_URLS, 1):
                print(f"[{i}/{total_models}] ", end="")
                sys.stdout.flush()
                if download_model(model_name, output_dir):
                    success_count += 1
            
            print(f"\nСкачано {success_count} из {total_models} моделей")
        else:
            download_model(args.model, output_dir)
    
    finally:
        # Восстанавливаем настройки терминала
        if terminal_configured:
            restore_terminal()

if __name__ == "__main__":
    main() 