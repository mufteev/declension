#!/usr/bin/env python3
"""
Скрипт для сравнения результатов сервиса склонений с эталонным тестовым набором.
"""

import json
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestCase:
    """Тестовый случай из markdown-таблицы."""
    number: int
    original: str
    target_case: str
    expected: str


@dataclass
class ServiceResult:
    """Результат от сервиса."""
    original: str
    result: str
    target_case: str
    target_case_readable: str
    confidence: float
    engine: str


@dataclass
class ComparisonResult:
    """Результат сравнения."""
    test_case: TestCase
    service_result: ServiceResult
    is_correct: bool
    

def parse_markdown_table(md_content: str) -> list[TestCase]:
    """Парсит markdown-таблицу с тестовыми данными."""
    test_cases = []
    lines = md_content.strip().split('\n')
    
    for line in lines:
        # Пропускаем заголовок и разделитель
        if not line.startswith('|') or '---' in line or '№' in line or 'Оригинальная' in line:
            continue
        
        # Разбиваем строку по |
        parts = [p.strip() for p in line.split('|')]
        # Убираем пустые элементы в начале и конце
        parts = [p for p in parts if p]
        
        if len(parts) >= 4:
            try:
                number = int(parts[0])
                original = parts[1]
                target_case = parts[2]
                expected = parts[3]
                
                test_cases.append(TestCase(
                    number=number,
                    original=original,
                    target_case=target_case,
                    expected=expected
                ))
            except ValueError:
                continue
    
    return test_cases


def parse_json_results(json_content: str) -> list[ServiceResult]:
    """Парсит JSON с результатами сервиса."""
    data = json.loads(json_content)
    results = []
    
    for item in data.get('results', []):
        results.append(ServiceResult(
            original=item.get('original', ''),
            result=item.get('result', ''),
            target_case=item.get('target_case', ''),
            target_case_readable=item.get('target_case_readable', ''),
            confidence=item.get('confidence', 0.0),
            engine=item.get('engine', '')
        ))
    
    return results


def normalize_case_name(case_name: str) -> str:
    """Нормализует название падежа для сравнения."""
    case_mapping = {
        # Полные русские названия
        'именительный': 'nomn',
        'родительный': 'gent',
        'дательный': 'datv',
        'винительный': 'accs',
        'творительный': 'ablt',
        'предложный': 'loct',
        # Сокращённые коды
        'nomn': 'nomn',
        'gent': 'gent',
        'datv': 'datv',
        'accs': 'accs',
        'ablt': 'ablt',
        'loct': 'loct',
    }
    return case_mapping.get(case_name.lower().strip(), case_name.lower().strip())


def compare_results(
    test_cases: list[TestCase], 
    service_results: list[ServiceResult]
) -> list[ComparisonResult]:
    """Сравнивает тестовые случаи с результатами сервиса."""
    comparisons = []
    
    for i, (test_case, service_result) in enumerate(zip(test_cases, service_results)):
        # Сравниваем ожидаемый результат с полученным
        is_correct = test_case.expected.strip().lower() == service_result.result.strip().lower()
        
        comparisons.append(ComparisonResult(
            test_case=test_case,
            service_result=service_result,
            is_correct=is_correct
        ))
    
    return comparisons


def print_statistics(comparisons: list[ComparisonResult]) -> None:
    """Выводит статистику сравнения."""
    total = len(comparisons)
    correct = sum(1 for c in comparisons if c.is_correct)
    incorrect = total - correct
    
    print("=" * 70)
    print("СТАТИСТИКА ТЕСТИРОВАНИЯ СЕРВИСА СКЛОНЕНИЙ")
    print("=" * 70)
    
    # Общая статистика
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   ✅ Правильно определённых: {correct} шт")
    print(f"   ❌ Неправильно определённых: {incorrect} шт")
    print(f"   📝 Всего тестов: {total} шт")
    
    # Процентная статистика
    if total > 0:
        success_rate = (correct / total) * 100
        print(f"\n📈 ПРОЦЕНТНАЯ СТАТИСТИКА:")
        print(f"   Точность: {success_rate:.2f}%")
        
        # Визуальная шкала
        bar_length = 40
        filled = int(bar_length * correct / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   [{bar}]")
    
    # Неправильные результаты
    incorrect_results = [c for c in comparisons if not c.is_correct]
    
    if incorrect_results:
        print(f"\n❌ НЕПРАВИЛЬНО ОПРЕДЕЛЁННЫЕ ({len(incorrect_results)} шт):")
        print("-" * 70)
        
        for comp in incorrect_results:
            print(f"\n   № {comp.test_case.number}:")
            print(f"   📝 Оригинал:    \"{comp.test_case.original}\"")
            print(f"   🎯 Целевой падеж: {comp.test_case.target_case} ({comp.service_result.target_case_readable})")
            print(f"   ✅ Ожидалось:  \"{comp.test_case.expected}\"")
            print(f"   ❌ Получено:   \"{comp.service_result.result}\"")
            print(f"   ⚙️  Engine:     {comp.service_result.engine} (confidence: {comp.service_result.confidence})")
    else:
        print("\n🎉 Все тесты пройдены успешно!")
    
    print("\n" + "=" * 70)


def main():
    """Основная функция."""
    
    if len(sys.argv) >= 3:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            md_content = f.read()
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            json_content = f.read()

    # Пример использования - замените на свои данные
    # Вариант 1: Чтение из файлов
    # with open('test_cases.md', 'r', encoding='utf-8') as f:
    #     md_content = f.read()
    # with open('results.json', 'r', encoding='utf-8') as f:
    #     json_content = f.read()

    # Парсинг данных
    test_cases = parse_markdown_table(md_content)
    service_results = parse_json_results(json_content)
    
    # Проверка количества
    if len(test_cases) != len(service_results):
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Количество тестов ({len(test_cases)}) "
              f"не совпадает с количеством результатов ({len(service_results)})")
        print(f"   Будут сравнены первые {min(len(test_cases), len(service_results))} записей.")
    
    # Сравнение
    comparisons = compare_results(test_cases, service_results)
    
    # Вывод статистики
    print_statistics(comparisons)


if __name__ == "__main__":
    main()