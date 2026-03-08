# Russian Declension API

Система склонения русских слов, словосочетаний и фраз.

## Быстрый старт

```bash
# Установка зависимостей
pip install pymorphy3 pydantic fastapi uvicorn[standard] natasha

# Запуск API-сервера
python -m russian_declension.main

# Или через uvicorn с hot-reload (для разработки)
uvicorn russian_declension.api.app:app --host 0.0.0.0 --port 8000 --reload



RUT5_MODEL_PATH=models/rut5-declension \
ANIMACY_MODEL_PATH=models/animacy_v1 \
ENSEMBLE_MODEL_PATH=models/ensemble \
GPU_DEVICE=cpu \
python -m russian_declension.main


RUT5_MODEL_PATH=models/rut5-declension ANIMACY_MODEL_PATH=models/animacy_v1 ENSEMBLE_MODEL_PATH=models/ensemble BERT_VALIDATOR_MODEL=models/bert GPU_DEVICE=auto python -m russian_declension.main
```

Swagger UI доступен на `http://localhost:8000/docs`.

## Примеры использования

### REST API

```bash
# Склонение слова
curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{"text": "документ", "target_case": "gent"}'

# Склонение фразы
curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{"text": "генеральный директор", "target_case": "datv"}'

# Склонение ФИО
curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{"text": "Иванов Иван Иванович", "target_case": "datv", "entity_type": "name", "gender": "male"}'

# Склонение организации (кавычечное правило)
curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{"text": "ООО «Ромашка»", "target_case": "gent", "entity_type": "org"}'

# Числительное с единицей измерения
curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{"text": "21 рубль", "target_case": "datv", "entity_type": "numeral"}'

# Полная парадигма слова (12 форм)
curl -X POST http://localhost:8000/api/v1/paradigm \
  -H "Content-Type: application/json" \
  -d '{"word": "документ"}'

# Пакетное склонение
curl -X POST http://localhost:8000/api/v1/inflect/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
    {"text": "директор", "target_case": "datv"},
    {"text": "Иванов Иван", "target_case": "gent", "entity_type": "name", "gender": "male"},
    {"text": "ООО «Ромашка»", "target_case": "gent", "entity_type": "org"}
  ]}'
```

### Как Python-библиотека

```python
from russian_declension.core.enums import Case, Number
from russian_declension.service import DeclensionService, EntityType

service = DeclensionService()

# Автоопределение типа
result = service.inflect("генеральный директор", Case.DATIVE)
print(result["result"])  # «генеральному директору»

# Явный тип
result = service.inflect("Иванов Иван Иванович", Case.GENITIVE,
                          entity_type=EntityType.NAME, gender="male")
print(result["result"])  # «Иванова Ивана Ивановича»

# Полная парадигма
paradigm = service.paradigm("кошка")
```

## Архитектура

```
Клиент → FastAPI → Маршрутизатор типов сущностей
                        │
          ┌─────────────┼─────────────┬──────────────┐
          ▼             ▼             ▼              ▼
     NameEngine    NumeralEngine  OrgEngine    PhraseEngine
     (ФИО)        (числительные) (организации) (dep.parsing)
          │             │             │              │
          └─────────────┴─────────────┴──────────────┘
                                │
                   FallbackChain (словоуровневый конвейер)
                        │
              ┌─────────┼──────────┐
              ▼         ▼          ▼
          LRU-кэш   pymorphy3   NeuralONNX
          (100K)    (400K лексем) (fallback)
```

## Структура проекта

```
russian_declension/
├── core/                  # Ядро: enums, модели, интерфейсы
│   ├── enums.py           # Case, Gender, Number, Animacy, SpecialGroup
│   ├── models.py          # MorphInfo, InflectionResult, FullParadigm
│   └── interfaces.py      # IDeclensionEngine, ICacheBackend
├── engines/               # Фаза 1: словоуровневый конвейер
│   ├── pymorphy_engine.py # Основной engine на pymorphy3
│   ├── cache.py           # LRU-кэш (100K записей, ~50 МБ)
│   └── fallback_chain.py  # Оркестратор «кэш → pymorphy → neural»
├── names/                 # Фаза 2: ФИО
│   └── engine.py          # Petrovich-стиль: фамилии, имена, отчества
├── numerals/              # Фаза 2: числительные
│   └── engine.py          # 7+ парадигм + согласование с существительным
├── organizations/         # Фаза 2: организации
│   └── engine.py          # Кавычечное правило + аббревиатуры
├── phrases/               # Фаза 3: фразовый движок
│   └── engine.py          # Natasha dep.parsing + agreement propagation
├── api/                   # REST API
│   ├── app.py             # FastAPI-приложение
│   └── schemas.py         # Pydantic-схемы запросов/ответов
├── service.py             # Центральный оркестратор (маршрутизация)
├── main.py                # Точка входа (uvicorn)
├── requirements.txt       # Зависимости
└── tests/
    └── demo.py            # Демонстрация + тесты всех трёх фаз
```

## Падежи (target_case)

| Код    | Падеж          | Вопрос           |
|--------|----------------|------------------|
| `nomn` | Именительный   | кто? что?        |
| `gent` | Родительный    | кого? чего?      |
| `datv` | Дательный      | кому? чему?      |
| `accs` | Винительный    | кого? что?       |
| `ablt` | Творительный   | кем? чем?        |
| `loct` | Предложный     | о ком? о чём?    |

## Типы сущностей (entity_type)

| Тип       | Описание                        | Engine           |
|-----------|---------------------------------|------------------|
| `auto`    | Автоопределение                 | эвристика        |
| `word`    | Одиночное слово                 | FallbackChain    |
| `name`    | ФИО                             | NameEngine       |
| `org`     | Название организации            | OrgEngine        |
| `numeral` | Числительное (+ед. измерения)   | NumeralEngine    |
| `phrase`  | Произвольная фраза              | PhraseEngine     |

## Запуск тестов

```bash
python -m russian_declension.tests.demo
```

```bash
curl -X POST http://localhost:8000/api/v1/inflect/batch \
  -H "Content-Type: application/json" \
  -d @test.json

curl -X POST http://localhost:8000/api/v1/inflect \
  -H "Content-Type: application/json" \
  -d '{ "text": "Инженер-программист", "target_case": "datv" }'
```