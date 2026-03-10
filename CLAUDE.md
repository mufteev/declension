# Russian Declension API

REST API и Python-библиотека для склонения русских слов, фраз, ФИО, числительных и организаций. CPU-режим (pymorphy3 + natasha) + опциональный GPU (ruT5, AnimacyClassifier, MetaEnsemble, BertValidator).

## Команды

### Окружение
- `python -m venv .venv && source .venv/bin/activate` — создать venv
- `pip install pymorphy3 pydantic fastapi "uvicorn[standard]" natasha` — CPU-зависимости
- `pip install torch --index-url https://download.pytorch.org/whl/cu124 && pip install transformers sentencepiece accelerate navec` — GPU-зависимости

### Данные для обучения
- `mkdir -p data && wget -O data/rus.tsv https://raw.githubusercontent.com/unimorph/rus/master/rus` — UniMorph Russian
- `wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar` — Navec embeddings

### Запуск сервера
- `python -m russian_declension.main` — CPU-only (порт 8000)
- `uvicorn russian_declension.api.app:app --host 0.0.0.0 --port 8000 --reload` — с hot-reload
- `RUT5_MODEL_PATH=models/rut5-declension ANIMACY_MODEL_PATH=models/animacy_v1 ENSEMBLE_MODEL_PATH=models/ensemble_v2 BERT_VALIDATOR_MODEL=models/rubert-base-cased-conversational_ner-v3 GPU_DEVICE=auto python -m russian_declension.main` — полный GPU-режим

### Обучение моделей (порядок важен)
- `python -m russian_declension.gpu.training.train_animacy --output models/animacy --epochs 20 --batch-size 256 --device auto` — AnimacyClassifier (~5 мин, CPU)
- `python russian_declension/gpu/training/animacy.py --navec-path data/navec_hudlit_v1_12B_500K_300d_100q.tar --opencorpora-xml data/dict.opcorpora.xml` — AnimacyClassifier с реальными данными (предпочтительно)
- `python -m russian_declension.gpu.training.train_ensemble --data data/rus.tsv --output models/ensemble --epochs 15 --batch-size 512 --max-samples 50000 --device auto` — MetaEnsemble (~10 мин)
- `python -m russian_declension.gpu.training.train_rut5 --data data/rus.tsv --base-model cointegrated/rut5-small --output models/rut5-declension --epochs 10 --batch-size 64 --lr 3e-4` — RuT5 (~30-60 мин, GPU)

### Тесты и демо
- `python -m russian_declension.tests.demo` — демонстрация всех компонентов

## Стек
- Python 3.x, PyTorch ≥2.0, transformers ≥4.30
- Морфологический движок: pymorphy3 ≥2.0 (основной), natasha ≥1.6 (dep.parsing)
- GPU-модели: ruT5-small (fine-tuned), AnimacyMLP (Navec 300d + suffix), EnsembleMLP (16 features → 3-way), rubert-base NER (validator)
- Данные: UniMorph Russian TSV, OpenCorpora XML, Navec embeddings
- Конфиги: env vars через `os.getenv()`
- Сервинг: FastAPI + uvicorn

## Архитектура проекта
```
russian_declension/
├── core/            # enums (Case/Gender/Number/Animacy), models (InflectionResult/FullParadigm/MorphInfo), interfaces (IDeclensionEngine/ICacheBackend)
├── engines/         # pymorphy_engine.py, cache.py (LRU 100K), fallback_chain.py (каскад/параллельный режим)
├── names/           # NameEngine — ФИО по частям (фамилия/имя/отчество)
├── numerals/        # NumeralEngine — числительные + согласование с сущ.
├── organizations/   # OrganizationEngine — кавычечное правило, аббревиатуры
├── phrases/         # PhraseEngine — natasha dep.parsing + agreement propagation
├── gpu/             # AnimacyClassifier, MetaEnsemble, RuT5Engine, BertValidator
│   └── training/    # train_animacy.py, train_ensemble.py, train_rut5.py, train_rut5_v2.py
├── api/             # FastAPI app.py + Pydantic schemas.py
├── service.py       # DeclensionService — центральный оркестратор
└── main.py          # точка входа uvicorn
```

Клиент → FastAPI → `DeclensionService.inflect()` → маршрутизация по `EntityType` → соответствующий Engine → `FallbackChain` (кэш → pymorphy → ruT5) → опционально GPU-постпроверка (BertValidator).

## ML Pipeline

1. **Данные**: UniMorph Russian TSV (лемма/форма/теги) + синтетика из pymorphy3; OpenCorpora XML для AnimacyClassifier
2. **AnimacyClassifier**: `[Navec(300d) | suffix_hash(30d)] → Linear(128) → ReLU → Dropout(0.3) → Linear(64) → Linear(1) → sigmoid`; сохраняется в `models/animacy/animacy_classifier.pt`
3. **MetaEnsemble (EnsembleMLP)**: `16 features → Linear(32) → ReLU → Dropout(0.2) → Linear(32) → Linear(3) → softmax`; feature vector строится идентично в train и runtime — критично не нарушать синхронизацию
4. **RuT5**: fine-tune `cointegrated/rut5-small` (или `ai-forever/FRED-T5-large`) seq2seq: prompt `"inflect: {lemma} case={code} number={code}"` → форма; сохраняется в `models/rut5-declension/`
5. **Инференс**: `FallbackChain.inflect()` каскадно или `inflect_all()` + `MetaEnsemble.select_best()`; AnimacyClassifier активируется при `Case.ACCUSATIVE` + `score < 0.5`; BertValidator при `confidence < 0.8`

## Конвенции кода
- Именование: `snake_case` файлы/функции/переменные, `PascalCase` классы, `UPPER_CASE` константы
- Импорты: абсолютные внутри пакета (`from .core.enums import Case`), `from __future__ import annotations` в модулях с forward-refs
- Type hints: да, повсеместно; `Optional[T]` вместо `T | None` (совместимость)
- Docstrings: минимальные, в телеграфном стиле; подробные комментарии блоками `# ──`
- Логирование: `logging.getLogger(__name__)`, `logger.info/debug/warning/error`
- Пути: `pathlib.Path` в training-скриптах, строки в runtime
- Конфиги: env vars через `os.getenv()` → конструктор `DeclensionService`
- Device: `"cuda" if torch.cuda.is_available() else "cpu"` при `device="auto"`

## Важные паттерны

**Новый Engine**: реализовать `IDeclensionEngine` (`core/interfaces.py`) — методы `inflect()`, `analyze()`, `paradigm()`, `healthcheck()`, свойство `name`, `confidence_threshold`; зарегистрировать в `DeclensionService.__init__()`, добавить в список `engines`.

**Feature vector MetaEnsemble**: порядок признаков в `train_ensemble.extract_features()` и `MetaEnsemble._extract_features()` должен быть **идентичен** — 16 значений: `[pymorphy×4, rut5×4, neural×4, suffix_3, len_1]`.

**GPU-компоненты lazy-load**: все GPU-классы инициализируются только при первом обращении к `is_available`. Безопасно создавать `DeclensionService` без GPU.

**FallbackChain режимы**: `inflect()` — каскад (первый уверенный); `inflect_all()` — параллельный сбор для MetaEnsemble.

**Кэш фраз**: `DeclensionService._phrase_cache` — dict в памяти, ключ `"text|case|entity_type|number|gender"`.

## Переменные окружения
- `RUT5_MODEL_PATH` — путь к директории fine-tuned ruT5 (e.g. `models/rut5-declension`)
- `ANIMACY_MODEL_PATH` — путь к директории AnimacyClassifier (e.g. `models/animacy_v1`)
- `ENSEMBLE_MODEL_PATH` — путь к директории MetaEnsemble (e.g. `models/ensemble_v2`)
- `BERT_VALIDATOR_MODEL` — путь к rubert NER-модели (e.g. `models/rubert-base-cased-conversational_ner-v3`)
- `GPU_DEVICE` — `auto` / `cuda` / `cpu` (default: `auto`)
- `CACHE_SIZE` — размер LRU-кэша (default: `100000`)

## Что НЕ делать
- Не коммитить `models/` и `data/` — они в `.gitignore`
- Не изменять порядок признаков в feature vector MetaEnsemble без синхронного обновления обоих файлов (`train_ensemble.py` и `ensemble.py`)
- Не использовать `FallbackChain.inflect_all()` без MetaEnsemble — это обходит кэш
- Не хардкодить device (`cuda`/`cpu`) — всегда через `GPU_DEVICE` env var или `device="auto"`
- `Case.NOMINATIVE` обрабатывается passthrough в `DeclensionService.inflect()` — не передавать через engine'ы
