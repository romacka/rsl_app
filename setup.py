#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для настройки приложения распознавания русского жестового языка
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def run_command(command):
    """Выполняет команду и возвращает результат"""
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Построчная буферизация
        )
        
        # Читаем вывод в реальном времени
        output = ""
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output += line
        
        # Ждем завершения процесса
        return_code = process.wait()
        
        # Проверяем код возврата
        if return_code != 0:
            error_output = process.stderr.read()
            return False, f"Ошибка: {error_output}"
        
        return True, output
    except Exception as e:
        return False, f"Ошибка при выполнении команды: {e}"

def install_requirements():
    """Устанавливает необходимые зависимости"""
    print("Установка зависимостей...")
    success, output = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    if success:
        print("✓ Зависимости успешно установлены")
    else:
        print("✗ Ошибка при установке зависимостей:")
        print(output)
    return success

def download_model(model_name="mvit32-2.onnx"):
    """Скачивает модель"""
    print(f"Скачивание модели {model_name}...")
    success, output = run_command(f"{sys.executable} download_models.py --model {model_name}")
    if success:
        print("✓ Модель успешно скачана")
    else:
        print("✗ Ошибка при скачивании модели:")
        print(output)
    return success

def extract_constants(constants_path):
    """Извлекает константы из constants.py"""
    if os.path.exists(constants_path):
        print("Извлечение словаря классов...")
        success, output = run_command(f"{sys.executable} extract_classes.py --input {constants_path} --output classes.json")
        if success:
            print("✓ Словарь классов успешно извлечен")
        else:
            print("✗ Ошибка при извлечении словаря классов:")
            print(output)
        return success
    else:
        print(f"✗ Файл constants.py не найден по пути {constants_path}")
        return False

def create_empty_classes_file():
    """Создает пустой файл classes.json"""
    try:
        if not os.path.exists("classes.json"):
            with open('classes.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            print("✓ Создан пустой файл classes.json")
            print("  Внимание: в приложении будут отображаться номера классов вместо названий жестов.")
        else:
            print("✓ Файл classes.json уже существует")
        return True
    except Exception as e:
        print(f"✗ Ошибка при создании файла classes.json: {e}")
        return False

def create_directories():
    """Создает необходимые директории"""
    print("Создание необходимых директорий...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    print("✓ Директории созданы")
    return True

def main():
    parser = argparse.ArgumentParser(description="Настройка приложения распознавания РЖЯ")
    parser.add_argument("--constants", help="Путь к файлу constants.py с объявлением словаря классов")
    parser.add_argument("--model", default="mvit32-2.onnx", 
                        choices=["mvit16-4.onnx", "mvit32-2.onnx", "mvit48-2.onnx",
                                "swin16-3.onnx", "swin32-2.onnx", "swin48-1.onnx",
                                "resnet16-3.onnx", "resnet32-2.onnx", "resnet48-1.onnx",
                                "SignFlow-A.onnx", "SignFlow-R.onnx", "all"],
                        help="Модель для скачивания (по умолчанию: mvit32-2.onnx)")
    parser.add_argument("--skip-deps", action="store_true", help="Пропустить установку зависимостей")
    
    args = parser.parse_args()
    
    # Приветствие
    print("=" * 70)
    print("Настройка приложения для распознавания русского жестового языка")
    print("=" * 70)
    
    # Создание директорий
    create_directories()
    
    # Установка зависимостей
    if not args.skip_deps:
        if not install_requirements():
            print("Предупреждение: Не удалось установить зависимости. Продолжаем...")
    else:
        print("Пропуск установки зависимостей (--skip-deps)")
    
    # Скачивание модели
    if not download_model(args.model):
        print("Предупреждение: Не удалось скачать модель. Продолжаем...")
    
    # Извлечение констант
    constants_extracted = False
    if args.constants:
        constants_extracted = extract_constants(args.constants)
    else:
        # Попытаемся найти constants.py в текущей или родительской директории
        potential_paths = ["constants.py", "../constants.py"]
        for path in potential_paths:
            if os.path.exists(path):
                print(f"Найден файл constants.py по пути {path}")
                constants_extracted = extract_constants(path)
                break
        
        if not constants_extracted:
            print("✗ Файл constants.py не найден автоматически.")
            print("  Укажите путь к файлу constants.py с помощью аргумента --constants")
    
    # Если не удалось извлечь константы, создаем пустой файл classes.json
    if not constants_extracted:
        create_empty_classes_file()
    
    # Готово
    print("\n" + "=" * 70)
    print("Настройка завершена!")
    print("Чтобы запустить приложение, выполните:")
    print(f"  {sys.executable} run.py")
    print("=" * 70)

if __name__ == "__main__":
    main() 