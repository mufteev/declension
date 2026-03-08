# Russian Declension API v2 — Setup & Deployment

## Обзор

Система работает в двух режимах: **CPU-only** (План 1, без GPU, все три фазы) и **CPU+GPU** (План 2, с RTX 5070 для повышения точности на OOV-словах).

В CPU-режиме система готова к работе после установки 4 пакетов. GPU-компоненты подключаются опционально через переменные окружения и требуют дополнительных шагов: скачивания базовой модели, fine-tuning на данных UniMorph, и экспорта модели.

---

## Быстрый старт (CPU-only, План 1)

Этот вариант покрывает все три фазы (слова, ФИО, числительные, организации, фразы) и достаточен для большинства сценариев шаблонов документов.

```bash
# 1. Установка зависимостей
pip install pymorphy3 pydantic fastapi "uvicorn[standard]" natasha

# 2. Запуск API-сервера
cd <директория-с-проектом>
python -m russian_declension.main
# API на http://localhost:8000, Swagger UI на http://localhost:8000/docs

# 3. Тестирование
python -m russian_declension.tests.demo
```

Natasha опциональна, но настоятельно рекомендуется: без неё фразовый движок (Фаза 3) использует упрощённую эвристику вместо полноценного dependency parsing.

---

## GPU-режим (План 2, RTX 5070)

### Шаг 1: Установка GPU-зависимостей

```bash
# PyTorch с CUDA (подберите версию CUDA под вашу систему)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# HuggingFace Transformers + утилиты
pip install transformers sentencepiece accelerate

# Navec embeddings (для AnimacyClassifier)
pip install navec

# Основные зависимости (если ещё не установлены)
pip install pymorphy3 pydantic fastapi "uvicorn[standard]" natasha
```

### Шаг 2: Подготовка данных для обучения ruT5

Ключевой датасет — **UniMorph Russian** (около 60K троек «лемма / словоформа / признаки»).

```bash
# Скачиваем UniMorph Russian
mkdir -p data
wget -O data/rus.tsv https://raw.githubusercontent.com/unimorph/rus/master/rus

# Проверяем формат (TSV: лемма <TAB> форма <TAB> признаки)
head -5 data/rus.tsv
# кошка	кошки	N;GEN;SG
# кошка	кошке	N;DAT;SG
# ...
```

Для расширения обучающей выборки можно сгенерировать синтетические данные из pymorphy3 (прогнать все лексемы словаря через все падежно-числовые комбинации). Скрипт `train_rut5.py` поддерживает флаг `--synthetic`.

### Шаг 3: Fine-tuning ruT5

Базовая модель — `cointegrated/rut5-small` (около 60M параметров). На RTX 5070 с 12 ГБ VRAM обучение занимает примерно 30–60 минут при batch_size=64.

```bash
python -m russian_declension.gpu.training.train_rut5 \
    --data data/rus.tsv \
    --base-model cointegrated/rut5-small \
    --output models/rut5-declension \
    --epochs 10 \
    --batch-size 64 \
    --lr 3e-4

# После обучения в models/rut5-declension/ появятся:
#   config.json, pytorch_model.bin, tokenizer.model, ...
```

Для экспериментов с более мощной моделью замените `cointegrated/rut5-small` на `ai-forever/FRED-T5-large` (потребуется ~1.5 ГБ VRAM, batch_size уменьшить до 16–32).

### Шаг 4: Запуск API с GPU-компонентами

GPU-компоненты активируются через переменные окружения. Если переменная не задана, соответствующий компонент просто не загружается, и система работает в CPU-режиме.

```bash
# Минимальный GPU-режим: только ruT5 для OOV-слов
RUT5_MODEL_PATH=models/rut5-declension \
python -m russian_declension.main

# Полный GPU-режим (все компоненты)
RUT5_MODEL_PATH=models/rut5-declension \
BERT_VALIDATOR_MODEL=models/bert-validator \
ANIMACY_MODEL_PATH=models/animacy \
ENSEMBLE_MODEL_PATH=models/ensemble \
GPU_DEVICE=cuda \
python -m russian_declension.main
```

### Шаг 5 (опционально): Обучение AnimacyClassifier

```python
# Подготовка данных: извлекаем anim/inan из OpenCorpora через pymorphy3
from pymorphy3 import MorphAnalyzer
import navec
import numpy as np

morph = MorphAnalyzer(lang="ru")
# ... извлечь слова с метками anim/inan, получить navec-эмбеддинги,
# обучить MLP-классификатор (2 слоя, 128 hidden), сохранить в models/animacy/
```

Детали реализации обучения AnimacyClassifier и MetaEnsemble оставлены для следующей итерации, так как они требуют экспериментальной настройки на реальных данных. В текущей версии оба компонента используют эвристические fallback'и, которые уже дают приемлемую точность.

---

## Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `CACHE_SIZE` | Размер LRU-кэша (записей) | `100000` |
| `RUT5_MODEL_PATH` | Путь к fine-tuned ruT5 | не задано (отключён) |
| `BERT_VALIDATOR_MODEL` | Путь к BERT-валидатору | не задано (отключён) |
| `ANIMACY_MODEL_PATH` | Путь к классификатору одушевлённости | не задано (отключён) |
| `ENSEMBLE_MODEL_PATH` | Путь к мета-ансамблю | не задано (отключён) |
| `GPU_DEVICE` | Устройство: `auto`, `cuda`, `cpu` | `auto` |

---

## Бюджет VRAM на RTX 5070 (12 ГБ)

| Компонент | VRAM (FP16) | Статус |
|---|---|---|
| ruT5-small | ~500 МБ | Реализован, нужен fine-tuning |
| BERT morpho-validator | ~500 МБ | Реализован (использует Natasha) |
| AnimacyClassifier | ~100 МБ | Реализован (эвристический fallback) |
| MetaEnsemble | ~10 МБ | Реализован (эвристический fallback) |
| CUDA overhead | ~500 МБ | — |
| **Итого** | **~1.6 ГБ** | 10+ ГБ свободно |

При переходе на FRED-T5-large добавится ~1 ГБ — всё ещё укладывается в 12 ГБ с запасом.

---

## Docker-деплоймент

```dockerfile
# CPU-only
FROM python:3.12-slim
WORKDIR /app
COPY russian_declension/ ./russian_declension/
RUN pip install pymorphy3 pydantic fastapi "uvicorn[standard]" natasha
EXPOSE 8000
CMD ["python", "-m", "russian_declension.main"]
```

```dockerfile
# GPU (NVIDIA)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY russian_declension/ ./russian_declension/
COPY models/ ./models/
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install transformers sentencepiece pymorphy3 pydantic fastapi "uvicorn[standard]" natasha
ENV RUT5_MODEL_PATH=/app/models/rut5-declension
ENV GPU_DEVICE=cuda
EXPOSE 8000
CMD ["python3", "-m", "russian_declension.main"]
```

---

## Что исправлено в v2 относительно v1

Восемь тестов, падавших в v1, теперь проходят:

**Фамилии (5 ошибок).** Корневая причина: метод `_apply_surname_ending` отрезал суффикс «-ов» перед добавлением окончания. `"Иванов"[:-2]` давало `"Иван"`, а `"Иван" + "а"` = `"Ивана"` вместо `"Иванова"`. Исправление: для мужских фамилий на -ов/-ев/-ин/-ын окончание теперь **добавляется к полной фамилии** (`"Иванов" + "а" = "Иванова"`). Для женских — отрезается только последняя буква «а» (`"Иванова"[:-1] = "Иванов"`, `"Иванов" + "ой" = "Ивановой"`).

**Предложные группы (3 ошибки).** Корневая причина: эвристический парсер не распознавал предлоги, и слова внутри предложных групп (например, «ограниченной» в «с ограниченной ответственностью») ошибочно помечались как модификаторы головного слова. Исправление: добавлен набор из 25+ русских предлогов; при обнаружении предлога всё после него до конца фразы помечается как предложная группа и не склоняется. Для Natasha-парсера добавлен рекурсивный обход nmod-поддеревьев — все зависимые генитивных/предложных модификаторов тоже блокируются от склонения.
