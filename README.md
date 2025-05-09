# Приложение для распознавания русского жестового языка

Приложение распознает жесты русского жестового языка с использованием предобученных моделей нейронных сетей.

## Требования

- Python 3.7+
- OpenCV
- PyQt5
- ONNX Runtime
- PyYAML
- openai

## Установка

### Вариант 1: Стандартная установка

1. Перейдите в директорию приложения:

```bash
cd RSL_app/
```

2. Запустите скрипт настройки для автоматической установки:

```bash
python setup.py
```

3. Скрипт автоматически создаст нужные директории, установит зависимости и скачает необходимые модели.

### Вариант 2: Установка через Conda (рекомендуется)

1. Если у вас не установлен Conda, [скачайте и установите Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Создайте новое Conda окружение:

```bash
# Создание окружения с Python 3.9
conda create -n rsl_app python=3.9
```

3. Активируйте окружение:

```bash
# В Linux/macOS
conda activate rsl_app

# В Windows
activate rsl_app
```

4. Установите зависимости:

```bash
# Обновление pip
pip install --upgrade pip

# Установка PyQt5 через conda
conda install -c conda-forge pyqt

# Установка остальных зависимостей через pip
pip install -r requirements.txt
```

5. Запустите скрипт настройки:

```bash
python setup.py
```

## Структура приложения

```
RSL_app/
│   ├── main.py            # Основной код приложения
│   ├── recognizer.py      # Модуль распознавания
│   ├── run.py             # Скрипт для запуска приложения
│   ├── setup.py           # Скрипт настройки
│   ├── requirements.txt   # Зависимости
│   ├── configs/           # Директория с конфигурационными файлами
│   │   └── config.yaml    # Пример конфигурации
│   ├── models/            # Директория для моделей нейронных сетей
│   │   └── *.onnx         # ONNX модели
│   ├── results/           # Директория для сохранения текстовых результатов
│   └── videos/            # Директория для сохранения видеозаписей
```

## Пример работы

Ниже показан пример работы приложения по распознаванию жестов:

![Пример работы](example/example.gif)

## Использование

1. Запустите приложение:

```bash
python run.py
```

2. В приложении:
   - Выберите модель нейронной сети из доступных в директории `models/`
   - Выберите конфигурационный файл
   - Нажмите кнопку "Запустить распознавание"
   - Показывайте жесты перед камерой
   - Результаты распознавания отобразятся в нижней части экрана
   - При необходимости записи видео нажмите кнопку "Записать видео"
   - После окончания работы сохраните результаты кнопкой "Сохранить результаты"

## Функции приложения

### Распознавание жестов
Основная функция приложения - распознавание жестов русского жестового языка в режиме реального времени с использованием выбранной нейронной сети.

### Сохранение результатов
Приложение позволяет сохранить распознанные жесты в текстовый файл. Файл содержит:
- Дату и время распознавания
- Использованную модель и конфигурацию
- Список всех распознанных жестов
- Последовательность жестов

### Запись видео
Приложение позволяет записывать видео с камеры с наложением распознанных жестов. Это полезно для:
- Документирования результатов
- Обучения и демонстрации
- Анализа точности распознавания

Файлы видео сохраняются в директории `videos/` в формате MP4.

## Конфигурация

Файл конфигурации (YAML) содержит следующие параметры:

```yaml
model_path: mvit32-2.onnx             # Имя файла модели
frame_interval: 2                      # Интервал между обрабатываемыми кадрами
mean: [123.675, 116.28, 103.53]        # Средние значения для нормализации изображения
std: [58.395, 57.12, 57.375]           # Стандартные отклонения для нормализации изображения
```

## Поддерживаемые модели

Приложение поддерживает следующие модели для распознавания жестов:

- MViT модели (mvit16-4.onnx, mvit32-2.onnx, mvit48-2.onnx)
- Swin модели (swin16-3.onnx, swin32-2.onnx, swin48-1.onnx)
- ResNet модели (resnet16-3.onnx, resnet32-2.onnx, resnet48-1.onnx)
- SignFlow модели (SignFlow-A.onnx, SignFlow-R.onnx)

Модели будут автоматически скачаны при запуске `setup.py`.

## Возможные проблемы и решения

1. **Модель не загружается**
   - Убедитесь, что модель находится в директории `models/`
   - Запустите скрипт `python setup.py` для повторного скачивания моделей

2. **Камера не открывается**
   - Проверьте, что камера подключена и работает
   - Проверьте права доступа к камере для приложения

3. **Ошибки при установке зависимостей**
   - Попробуйте установить зависимости вручную:
     ```bash
     pip install numpy opencv-python PyQt5 onnxruntime PyYAML
     ```
   
4. **Проблемы с OpenCV в Conda**
   - Установите OpenCV через conda:
     ```bash
     conda install -c conda-forge opencv
     ```

