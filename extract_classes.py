#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для извлечения словаря классов из constants.py
"""

import os
import re
import json
import argparse

def extract_classes(constants_file, output_file=None):
    """
    Извлекает словарь классов из файла constants.py и сохраняет его в JSON
    
    Args:
        constants_file (str): Путь к файлу constants.py
        output_file (str, optional): Путь для сохранения результата в JSON. 
                                    Если None, выводит на экран
    
    Returns:
        dict: Словарь классов (индекс -> название)
    """
    classes_dict = {}
    
    try:
        with open(constants_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Метод 1: Ищем блок определения словаря classes
        match = re.search(r'classes\s*=\s*{([^}]+)}', content, re.DOTALL)
        if match:
            classes_block = match.group(1)
            # Извлекаем пары ключ-значение
            pairs = re.findall(r'(\d+):\s*"([^"]+)"', classes_block)
            for idx, label in pairs:
                classes_dict[int(idx)] = label
        
        # Если первый метод не нашел классы или нашел мало, пробуем альтернативный подход
        if not classes_dict:
            print("Словарь classes не найден стандартным методом, пробуем альтернативный...")
            
            # Метод 2: Ищем определение словаря через отдельные присваивания
            pattern = r'classes\[(\d+)\]\s*=\s*"([^"]+)"'
            pairs = re.findall(pattern, content)
            for idx, label in pairs:
                classes_dict[int(idx)] = label
            
            # Метод 3: Ищем паттерны с номерами классов и метками
            if not classes_dict:
                print("Пробуем найти классы по паттерну ключ-значение...")
                # Ищем любые паттерны вида '123: "Название жеста"' или 'idx: "Название жеста"'
                pattern = r'(?:(?:[\'"]\s*|^|[,{]\s*)(\d+)\s*[:\']|\[(\d+)\])\s*(?::|=)\s*[\'"](.*?)[\'"]'
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    # Номер класса может быть в первой или второй группе
                    idx = match[0] if match[0] else match[1]
                    label = match[2].strip()
                    if idx and label:
                        classes_dict[int(idx)] = label
                
        if not classes_dict:
            print("Ошибка: Не удалось найти словарь классов в файле")
            # Создаем базовый словарь с несколькими классами
            for i in range(10):
                classes_dict[i] = f"Класс {i}"
            print("Создан базовый словарь с классами от 0 до 9")
        else:
            print(f"Найдено {len(classes_dict)} классов")
            
    except Exception as e:
        print(f"Ошибка при чтении файла constants.py: {e}")
        return {}
    
    # Сохраняем или выводим результат
    if output_file:
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(classes_dict, f, ensure_ascii=False, indent=4)
            print(f"Словарь классов сохранен в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")
    else:
        print(json.dumps(classes_dict, ensure_ascii=False, indent=4))
    
    return classes_dict

def main():
    parser = argparse.ArgumentParser(description="Извлечение словаря классов из constants.py")
    parser.add_argument("--input", required=True, help="Путь к файлу constants.py")
    parser.add_argument("--output", help="Путь для сохранения словаря в JSON (по умолчанию: classes.json)")
    parser.add_argument("--force", action="store_true", help="Перезаписать существующий файл, если он есть")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output or "classes.json"
    
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден")
        return
    
    # Проверяем, существует ли файл и нужно ли его перезаписать
    if os.path.exists(output_file) and not args.force:
        response = input(f"Файл {output_file} уже существует. Перезаписать? (y/n): ")
        if response.lower() != 'y':
            print("Операция отменена")
            return
    
    extract_classes(input_file, output_file)

if __name__ == "__main__":
    main() 