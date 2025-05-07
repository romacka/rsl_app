#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для создания пустого файла classes.json, если он отсутствует
"""

import os
import json
import sys

def create_empty_classes_file(output_file="classes.json"):
    """
    Создает пустой файл classes.json, если он отсутствует
    
    Args:
        output_file (str): Путь к файлу classes.json
    """
    if os.path.exists(output_file):
        print(f"Файл {output_file} уже существует.")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
        print(f"Создан пустой файл {output_file}")
        print("Внимание: в приложении будут отображаться номера классов вместо названий жестов.")
    except Exception as e:
        print(f"Ошибка при создании файла: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_empty_classes_file(sys.argv[1])
    else:
        create_empty_classes_file() 