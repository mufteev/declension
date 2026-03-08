"""
# Вариант 1: с файлами по умолчанию (input.md → output.json)
python prepare_case_to_input.py

# Вариант 2: указать входной файл (→ output.json)
python prepare_case_to_input.py test_cases.md

# Вариант 3: указать оба файла
python prepare_case_to_input.py test_cases.md result.json
"""

import json
import re
import sys



def parse_md_table_to_json(md_table: str) -> dict:
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    
    # Пропускаем заголовок и разделитель
    data_lines = [line for line in lines[2:] if not line.startswith('|-')]
    
    # Регулярка для извлечения кода падежа из скобок
    case_pattern = re.compile(r'\((\w+)\)')
    
    items = []
    seen_phrases = set()
    
    for line in data_lines:
        # Разбиваем строку по |
        cells = [cell.strip() for cell in line.split('|')]
        # Убираем пустые элементы по краям
        cells = [c for c in cells if c]
        
        if len(cells) >= 3:
            original_phrase = cells[1].strip()
            target_case_full = cells[2].strip()
            
            # Извлекаем код падежа
            match = case_pattern.search(target_case_full)
            if match:
                target_case = match.group(1)
                
                # Добавляем именительный падеж (nomn) для первого вхождения фразы
                if original_phrase not in seen_phrases:
                    items.append({
                        "text": original_phrase,
                        "target_case": "nomn"
                    })
                    seen_phrases.add(original_phrase)
                
                # Добавляем текущий падеж
                items.append({
                    "text": original_phrase,
                    "target_case": target_case
                })
    
    return {"items": items}


def main():
    # Имена файлов по умолчанию или из аргументов командной строки
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = "output.json"
    else:
        input_file = "input.md"
        output_file = "output.json"
    
    # Чтение из файла
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            md_table = f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit(1)
    
    # Парсинг
    result = parse_md_table_to_json(md_table)
    
    # Запись в файл
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка при записи файла: {e}")
        sys.exit(1)
    
    print(f"Готово! Обработано {len(result['items'])} записей")
    print(f"Результат сохранён в '{output_file}'")


if __name__ == "__main__":
    main()
