import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import shutil

def analyze_pdf_text_layer(pdf_path):
    """
    Анализирует текстовый слой PDF с помощью PyMuPDF
    
    Возвращает:
        dict: результаты анализа текстового слоя
    """
    print("📖 Анализируем текстовый слой...")
    
    doc = fitz.open(pdf_path)
    results = {
        'has_text_layer': False,
        'total_pages': len(doc),
        'pages_with_text': 0,
        'total_text_length': 0,
        'text_pages_ratio': 0.0,
        'avg_text_per_page': 0.0
    }
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if text.strip():  # Если на странице есть текст
            results['has_text_layer'] = True
            results['pages_with_text'] += 1
            results['total_text_length'] += len(text.strip())
    
    # Расчет метрик
    if results['total_pages'] > 0:
        results['text_pages_ratio'] = results['pages_with_text'] / results['total_pages']
        results['avg_text_per_page'] = results['total_text_length'] / max(results['pages_with_text'], 1)
    
    doc.close()
    return results

def calculate_text_density(image):
    """
    Рассчитывает плотность текста на изображении через бинаризацию
    
    Возвращает:
        float: плотность текста (0-1)
    """
    try:
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        else:
            gray = np.array(image)
        
        # Бинаризация (черный/белый)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Подсчет черных пикселей (предполагаемый текст)
        black_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        text_density = black_pixels / total_pixels
        return min(text_density, 1.0)  # ограничиваем 1.0
        
    except Exception as e:
        print(f"⚠ Ошибка анализа плотности: {e}")
        return 0.0

def analyze_visual_text_density(pdf_path, sample_pages=5):
    """
    Анализирует визуальную плотность текста на нескольких страницах
    
    Возвращает:
        float: средняя плотность текста
    """
    print("👁 Анализируем визуальную плотность...")
    
    try:
        # Конвертируем PDF в изображения (только несколько страниц для скорости)
        images = convert_from_path(pdf_path, first_n=min(sample_pages, 10), dpi=100)
        
        if not images:
            return 0.0
        
        densities = []
        for img in images:
            density = calculate_text_density(img)
            densities.append(density)
        
        avg_density = np.mean(densities) if densities else 0.0
        return avg_density
        
    except Exception as e:
        print(f"⚠ Ошибка визуального анализа: {e}")
        return 0.0

def get_file_metrics(pdf_path):
    """
    Собирает файловые метрики
    
    Возвращает:
        dict: файловые характеристики
    """
    file_stats = os.stat(pdf_path)
    return {
        'file_size_mb': file_stats.st_size / (1024 * 1024),
        'filename': os.path.basename(pdf_path)
    }

def needs_ocr_analysis(text_analysis, visual_analysis, file_metrics):
    """
    Принимает решение о необходимости OCR
    
    Возвращает:
        bool: True если нужен OCR, False если нет
    """
    # Правила определения необходимости OCR
    rules = [
        # Нет текстового слоя вообще
        not text_analysis['has_text_layer'],
        
        # Меньше 30% страниц содержат текст
        text_analysis['text_pages_ratio'] < 0.3,
        
        # Низкая плотность текста (меньше 2%)
        visual_analysis < 0.02,
        
        # Большой файл но мало текста (возможно сканы)
        (file_metrics['file_size_mb'] > 5 and 
         text_analysis['avg_text_per_page'] < 50),
        
        # Много страниц но мало текста
        (text_analysis['total_pages'] > 10 and 
         text_analysis['text_pages_ratio'] < 0.5)
    ]
    
    # Если хотя бы одно правило выполняется - нужен OCR
    return any(rules)

def setup_poppler():
    """Находит Poppler в системе"""
    poppler_cmd = shutil.which("pdftoppm")
    if poppler_cmd:
        return os.path.dirname(poppler_cmd)
    return None

def analyze_pdf_ocr_need(pdf_path):
    """
    Основная функция анализа необходимости OCR
    
    Возвращает:
        dict: результаты анализа
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
        
        # 4. Принятие решения
        ocr_required = needs_ocr_analysis(text_analysis, visual_density, file_metrics)
        
        # 5. Формируем результат
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
        print(f"❌ Ошибка анализа: {e}")
        return None

# Пример использования
if __name__ == "__main__":
    # Тестовый вызов - замени путь на свой PDF
    pdf_path = r"C:\Users\Pavel\Desktop\Dev\FirstTask\testpdf1.pdf"  # укажи путь к своему PDF
    
    if os.path.exists(pdf_path):
        result = analyze_pdf_ocr_need(pdf_path)
        if result:
            print("\n📋 ФИНАЛЬНЫЙ ОТЧЕТ:")
            print(f"Файл: {result['filename']}")
            print(f"Текстовый слой: {'Есть' if result['has_text_layer'] else 'Нет'}")
            print(f"Доля текстовых страниц: {result['text_pages_ratio']:.1%}")
            print(f"Плотность текста: {result['avg_text_density']:.3%}")
            print(f"OCR required: {result['ocr_required']}")
    else:
        print("❌ PDF файл не найден!")