#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для запуска приложения распознавания жестов РЖЯ
"""

import os
import sys
import subprocess

def setup():
    """Проверка и установка необходимых зависимостей"""
    try:
        # Проверяем наличие директорий для моделей и конфигураций
        os.makedirs('models', exist_ok=True)
        os.makedirs('configs', exist_ok=True)
        
        # Проверяем, что файл конфигурации существует
        if not os.path.exists('configs/config.yaml'):
            print("Файл конфигурации не найден!")
            return False
        
        # Проверяем, есть ли модели в директории
        models = [f for f in os.listdir('models') if f.endswith('.onnx')]
        if not models:
            print("В директории 'models/' не найдено моделей (.onnx)!")
            print("Запустите 'python setup.py' для загрузки моделей")
            return False
        
        return True
    except Exception as e:
        print(f"Ошибка при настройке: {e}")
        return False

def main():
    """Запуск приложения"""
    # Проверяем установку перед запуском
    if not setup():
        print("\nПриложение не может быть запущено. Пожалуйста, проверьте наличие моделей и конфигураций.")
        return
    
    try:
        # Запускаем основной скрипт
        print("Запуск приложения распознавания жестов РЖЯ...")
        import main
        from PyQt5.QtWidgets import QApplication
        
        # Сначала создаем QApplication, затем виджеты
        app = QApplication(sys.argv)
        window = main.RSLRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except ModuleNotFoundError as e:
        module = str(e).split("'")[1]
        print(f"Ошибка: Модуль {module} не найден!")
        print("Установка зависимостей...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Зависимости установлены. Повторный запуск приложения...")
            main()
        except Exception as e:
            print(f"Ошибка при установке зависимостей: {e}")
    except Exception as e:
        print(f"Ошибка при запуске приложения: {e}")

if __name__ == "__main__":
    main() 