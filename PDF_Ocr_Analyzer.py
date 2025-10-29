import tkinter as tk  # для создания графического интерфейса
from tkinter import filedialog, messagebox  # диалоговые окна выбора файлов и сообщения
import fitz  # PyMuPDF - для работы с PDF, анализа текстового слоя
import os  # для работы с файловой системой, путями
import cv2  # OpenCV - для обработки изображений, бинаризации
import numpy as np  # для численных операций, работы с массивами
from pdf2image import convert_from_path  # конвертация PDF в изображения
import shutil  # для поиска исполняемых файлов в системе
import glob  # для поиска файлов по шаблону
import csv  # для сохранения результатов в CSV формат

def setup_poppler():
    """
    Находит Poppler в системном PATH для конвертации PDF в изображения
    
    Возвращает:
        str или None: путь к папке bin Poppler или None если не найден
    """
    # Ищем исполняемый файл pdftoppm в системном PATH
    poppler_cmd = shutil.which("pdftoppm")
    if poppler_cmd:
        # Возвращаем папку, где находится исполняемый файл
        poppler_dir = os.path.dirname(poppler_cmd)
        print(f"✓ Poppler найден: {poppler_dir}")
        return poppler_dir
    else:
        print("❌ Poppler не найден в PATH")
        print("   Установите Poppler и добавьте папку bin в переменную PATH")
        return None

def analyze_pdf_text_layer(pdf_path):
    """
    Анализирует текстовый слой PDF документа с помощью PyMuPDF
    
    Анализирует:
    - Наличие любого текста в документе
    - Количество страниц с текстом
    - Общее количество символов текста
    - Соотношение текстовых/нетекстовых страниц
    
    Аргументы:
        pdf_path (str): путь к PDF файлу
        
    Возвращает:
        dict: словарь с результатами анализа текстового слоя
    """
    print("📖 Анализируем текстовый слой...")
    
    # Открываем PDF документ
    doc = fitz.open(pdf_path)
    
    # Инициализируем словарь для результатов
    results = {
        'has_text_layer': False,  # есть ли хоть какой-то текст в документе
        'total_pages': len(doc),  # общее количество страниц
        'pages_with_text': 0,  # количество страниц с текстом
        'total_text_length': 0,  # общее количество символов текста
        'text_pages_ratio': 0.0,  # доля страниц с текстом от общего числа
        'avg_text_per_page': 0.0  # среднее количество символов на текстовой странице
    }
    
    # Анализируем каждую страницу документа
    for page_num in range(len(doc)):
        page = doc[page_num]  # получаем объект страницы
        text = page.get_text()  # извлекаем весь текст со страницы
        
        # Если на странице есть непустой текст
        if text.strip():
            results['has_text_layer'] = True  # отмечаем что в документе есть текст
            results['pages_with_text'] += 1  # увеличиваем счетчик текстовых страниц
            results['total_text_length'] += len(text.strip())  # суммируем длину текста
    
    # Рассчитываем производные метрики
    if results['total_pages'] > 0:
        # Доля страниц с текстом = текстовые страницы / все страницы
        results['text_pages_ratio'] = results['pages_with_text'] / results['total_pages']
        # Средний текст на страницу = общий текст / текстовые страницы (минимум 1 чтобы избежать деления на 0)
        results['avg_text_per_page'] = results['total_text_length'] / max(results['pages_with_text'], 1)
    
    # Закрываем документ чтобы освободить память
    doc.close()
    return results

def calculate_text_density(image):
    """
    Рассчитывает визуальную плотность текста на изображении через бинаризацию
    
    Процесс:
    1. Конвертация PIL Image в numpy array
    2. Преобразование в оттенки серого (если нужно)
    3. Бинаризация для разделения на черный (текст) и белый (фон)
    4. Подсчет соотношения черных пикселей ко всем пикселям
    
    Аргументы:
        image: PIL Image объект
        
    Возвращает:
        float: плотность текста от 0.0 (нет текста) до 1.0 (все пиксели черные)
    """
    try:
        # Конвертируем PIL Image в numpy array для работы с OpenCV
        img_array = np.array(image)
        
        # Если изображение цветное (3 канала), конвертируем в grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array  # уже в оттенках серого
        
        # Бинаризация - преобразование в черно-белое изображение
        # THRESH_BINARY_INV - инвертирует, чтобы текст был белым (255), а фон черным (0)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Подсчет черных пикселей (в инвертированном изображении это бывший текст)
        black_pixels = np.sum(binary == 255)  # считаем пиксели со значением 255 (белые)
        total_pixels = binary.shape[0] * binary.shape[1]  # общее количество пикселей
        
        # Плотность текста = черные пиксели / все пиксели
        text_density = black_pixels / total_pixels
        
        # Ограничиваем максимальное значение 1.0
        return min(text_density, 1.0)
        
    except Exception as e:
        # В случае ошибки возвращаем 0 и логируем ошибку
        print(f"⚠ Ошибка анализа плотности: {e}")
        return 0.0

def analyze_visual_text_density(pdf_path, sample_pages=5):
    """
    Анализирует визуальную плотность текста на нескольких случайных страницах PDF
    
    Аргументы:
        pdf_path (str): путь к PDF файлу
        sample_pages (int): количество страниц для анализа (по умолчанию 5)
        
    Возвращает:
        float: средняя плотность текста на анализируемых страницах
    """
    print("👁 Анализируем визуальную плотность...")
    
    try:
        # Конвертируем PDF в изображения (только первые N страниц для скорости)
        images = convert_from_path(
            pdf_path, 
            first_page=1, 
            last_page=min(sample_pages, 10),  # анализируем не более 10 страниц
            dpi=100  # низкое разрешение для скорости, достаточно для анализа плотности
        )
        
        # Если не удалось получить изображения
        if not images:
            return 0.0
        
        # Список для хранения плотности каждой страницы
        densities = []
        
        # Анализируем каждое изображение
        for i, img in enumerate(images):
            density = calculate_text_density(img)
            densities.append(density)
            print(f"   Страница {i+1}: плотность {density:.3%}")
        
        # Рассчитываем среднюю плотность по всем анализируемым страницам
        avg_density = np.mean(densities) if densities else 0.0
        print(f"   Средняя плотность: {avg_density:.3%}")
        
        return avg_density
        
    except Exception as e:
        # В случае ошибки логируем и возвращаем 0
        print(f"⚠ Ошибка визуального анализа: {e}")
        return 0.0

def get_file_metrics(pdf_path):
    """
    Собирает базовые метрики файла
    
    Аргументы:
        pdf_path (str): путь к PDF файлу
        
    Возвращает:
        dict: словарь с файловыми характеристиками
    """
    # Получаем статистику файла
    file_stats = os.stat(pdf_path)
    
    return {
        'file_size_mb': file_stats.st_size / (1024 * 1024),  # размер в мегабайтах
        'filename': os.path.basename(pdf_path)  # имя файла без пути
    }

def needs_ocr_analysis(text_analysis, visual_analysis, file_metrics):
    """
    Принимает решение о необходимости OCR на основе анализа метрик
    
    Правила определения необходимости OCR:
    1. Нет текстового слоя вообще
    2. Меньше 30% страниц содержат текст
    3. Низкая визуальная плотность текста (<2%)
    4. Большой файл но мало текста (возможно сканы)
    5. Много страниц но мало текста
    
    Аргументы:
        text_analysis (dict): результаты анализа текстового слоя
        visual_analysis (float): визуальная плотность текста
        file_metrics (dict): файловые метрики
        
    Возвращает:
        bool: True если требуется OCR, False если нет
    """
    # Правила определения необходимости OCR
    rules = [
        # Правило 1: Нет текстового слоя вообще
        not text_analysis['has_text_layer'],
        
        # Правило 2: Меньше 30% страниц содержат текст
        text_analysis['text_pages_ratio'] < 0.3,
        
        # Правило 3: Низкая плотность текста (меньше 2%)
        visual_analysis < 0.02,
        
        # Правило 4: Большой файл но мало текста (возможно сканы)
        (file_metrics['file_size_mb'] > 5 and 
         text_analysis['avg_text_per_page'] < 50),
        
        # Правило 5: Много страниц но мало текста
        (text_analysis['total_pages'] > 10 and 
         text_analysis['text_pages_ratio'] < 0.5)
    ]
    
    # Если хотя бы одно правило выполняется - нужен OCR
    return any(rules)

def analyze_pdf_ocr_need(pdf_path):
    """
    Основная функция анализа необходимости OCR для одного PDF файла
    
    Процесс:
    1. Анализ файловых метрик
    2. Анализ текстового слоя через PyMuPDF  
    3. Визуальный анализ плотности текста
    4. Принятие решения о необходимости OCR
    
    Аргументы:
        pdf_path (str): путь к PDF файлу
        
    Возвращает:
        dict: полные результаты анализа или None при ошибке
    """
    print(f"🔍 Анализируем PDF: {os.path.basename(pdf_path)}")
    print("=" * 50)
    
    try:
        # 1. Файловые метрики
        file_metrics = get_file_metrics(pdf_path)
        print(f"📊 Размер файла: {file_metrics['file_size_mb']:.2f} MB")
        
        # 2. Анализ текстового слоя
        text_analysis = analyze_pdf_text_layer(pdf_path)
        print(f"📖 Страниц с текстом: {text_analysis['pages_with_text']}/{text_analysis['total_pages']}")
        print(f"📊 Доля текстовых страниц: {text_analysis['text_pages_ratio']:.1%}")
        print(f"📝 Средний текст на страницу: {text_analysis['avg_text_per_page']:.0f} символов")
        
        # 3. Визуальный анализ (только если Poppler доступен)
        poppler_path = setup_poppler()
        if poppler_path:
            visual_density = analyze_visual_text_density(pdf_path)
            print(f"🎯 Визуальная плотность текста: {visual_density:.3%}")
        else:
            visual_density = 0.0
            print("⚠ Poppler не найден, пропускаем визуальный анализ")
        
        # 4. Принятие решения о необходимости OCR
        ocr_required = needs_ocr_analysis(text_analysis, visual_density, file_metrics)
        
        # 5. Формируем итоговый результат
        result = {
            'filename': file_metrics['filename'],
            'has_text_layer': text_analysis['has_text_layer'],
            'text_pages_ratio': text_analysis['text_pages_ratio'],
            'avg_text_density': visual_density,
            'ocr_required': ocr_required,
            'file_size_mb': file_metrics['file_size_mb'],
            'total_pages': text_analysis['total_pages']
        }
        
        print("=" * 50)
        status = "🚨 OCR ТРЕБУЕТСЯ" if ocr_required else "✅ OCR НЕ ТРЕБУЕТСЯ"
        print(f"РЕЗУЛЬТАТ: {status}")
        
        return result
        
    except Exception as e:
        # Обработка критических ошибок
        print(f"❌ Ошибка анализа: {e}")
        return None

def select_folder():
    """
    Открывает диалоговое окно для выбора папки с PDF файлами
    
    Возвращает:
        str или None: путь к выбранной папке или None если папка не выбрана
    """
    # Создаем скрытое окно tkinter
    root = tk.Tk()
    root.withdraw()  # скрываем главное окно
    
    # Показываем диалог выбора папки
    folder_path = filedialog.askdirectory(title="Выберите папку с PDF файлами")
    return folder_path

def find_pdf_files(folder_path):
    """
    Рекурсивно находит все PDF файлы в папке и ее подпапках
    
    Аргументы:
        folder_path (str): путь к корневой папке для поиска
        
    Возвращает:
        list: список полных путей к PDF файлам
    """
    # **/*.pdf - рекурсивный поиск всех PDF файлов
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    return pdf_files

def analyze_pdf_folder(folder_path):
    """
    Анализирует все PDF файлы в указанной папке
    
    Аргументы:
        folder_path (str): путь к папке с PDF файлами
        
    Возвращает:
        list: список результатов анализа для каждого файла
    """
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        print("❌ Папка не существует!")
        return []
    
    # Находим все PDF файлы
    pdf_files = find_pdf_files(folder_path)
    
    if not pdf_files:
        print("❌ PDF файлы не найдены в папке!")
        return []
    
    print(f"📁 Найдено PDF файлов: {len(pdf_files)}")
    print("=" * 60)
    
    results = []  # список для хранения результатов
    
    # Обрабатываем каждый файл
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📄 Обрабатываем файл {i}/{len(pdf_files)}:")
        print(f"   {os.path.basename(pdf_path)}")
        
        try:
            # Анализируем текущий PDF файл
            result = analyze_pdf_ocr_need(pdf_path)
            if result:
                results.append(result)
        except Exception as e:
            # В случае ошибки добавляем базовую информацию
            print(f"❌ Ошибка обработки файла: {e}")
            results.append({
                'filename': os.path.basename(pdf_path),
                'has_text_layer': False,
                'text_pages_ratio': 0.0,
                'avg_text_density': 0.0,
                'ocr_required': True,  # при ошибке предполагаем что OCR нужен
                'file_size_mb': os.path.getsize(pdf_path) / (1024 * 1024),
                'total_pages': 0,
                'error': str(e)  # сохраняем сообщение об ошибке
            })
    
    return results

def print_summary_report(results):
    """
    Печатает красивый сводный отчет по всем обработанным файлам
    
    Аргументы:
        results (list): список результатов анализа
    """
    print("\n" + "=" * 80)
    print("📊 СВОДНЫЙ ОТЧЕТ ПО ВСЕМ ФАЙЛАМ")
    print("=" * 80)
    
    # Статистика
    total_files = len(results)
    ocr_required_count = sum(1 for r in results if r.get('ocr_required', False))
    has_text_count = sum(1 for r in results if r.get('has_text_layer', False))
    
    print(f"📁 Всего файлов: {total_files}")
    print(f"🔤 Файлов с текстовым слоем: {has_text_count}")
    print(f"🎯 Требуют OCR: {ocr_required_count}")
    print(f"✅ Не требуют OCR: {total_files - ocr_required_count}")
    print("-" * 80)
    
    # Сортируем файлы: сначала требующие OCR, потом остальные
    results_sorted = sorted(results, key=lambda x: (not x.get('ocr_required', False), x['filename']))
    
    # Выводим детали по каждому файлу
    for result in results_sorted:
        status = "🚨 OCR ТРЕБУЕТСЯ" if result.get('ocr_required', False) else "✅ OCR НЕ НУЖЕН"
        error_mark = " ⚠ ОШИБКА" if result.get('error') else ""
        
        print(f"{result['filename']:<40} {status}{error_mark}")
        
        if result.get('error'):
            # Для файлов с ошибками показываем сообщение об ошибке
            print(f"   Ошибка: {result['error']}")
        else:
            # Для успешно обработанных файлов показываем детали
            print(f"   Текстовый слой: {'Есть' if result['has_text_layer'] else 'Нет'}")
            print(f"   Плотность текста: {result.get('avg_text_density', 0):.3%}")
            print(f"   Страниц с текстом: {result.get('text_pages_ratio', 0):.1%}")

def save_to_csv(results, filename):
    """
    Сохраняет результаты анализа в CSV файл для дальнейшего использования
    
    Аргументы:
        results (list): список результатов анализа
        filename (str): имя CSV файла для сохранения
    """
    if not results:
        print("❌ Нет данных для сохранения!")
        return
    
    # Определяем поля для CSV файла
    fieldnames = ['filename', 'has_text_layer', 'avg_text_density', 'ocr_required']
    
    try:
        # Открываем файл для записи
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # записываем заголовок
            
            # Записываем данные для каждого файла
            for result in results:
                # Создаем строку только с нужными полями
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"💾 Результаты сохранены в: {filename}")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения в CSV: {e}")

def main():
    """
    Главная функция программы - точка входа
    """
    print("🎯 Анализатор необходимости OCR для PDF файлов")
    print("=" * 50)
    
    # 1. Выбор папки пользователем
    folder_path = select_folder()
    if not folder_path:
        print("❌ Папка не выбрана!")
        return False
    
    # 2. Анализ всех PDF файлов в папке
    results = analyze_pdf_folder(folder_path)
    
    if results:
        # 3. Вывод сводного отчета в консоль
        print_summary_report(results)
        
        # 4. Сохранение результатов в CSV файл
        save_to_csv(results, "ocr_analysis_report.csv")
        
        # 5. Показываем итоговое сообщение
        ocr_required_count = sum(1 for r in results if r.get('ocr_required', False))
        messagebox.showinfo("Анализ завершен", 
                          f"Обработано файлов: {len(results)}\n"
                          f"Требуют OCR: {ocr_required_count}")
    else:
        messagebox.showwarning("Нет результатов", "Не удалось обработать ни один файл!")
    
    return True

# Точка входа в программу
if __name__ == "__main__":
    # Запускаем главную функцию
    success = main()
    
    # Сообщаем о результате выполнения
    if success:
        print("\n✅ Программа завершена успешно!")
    else:
        print("\n❌ Программа завершена с ошибками!")