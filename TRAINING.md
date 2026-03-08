# Обучение GPU-моделей: полное руководство

Все три GPU-модели обучаются независимо друг от друга. Каждую можно обучить и подключить отдельно — система продолжит работать в CPU-режиме для компонентов без обученных моделей.

---

## Порядок обучения (рекомендуемый)

```
1. AnimacyClassifier  (~5 минут, не требует данных из интернета)
2. MetaEnsemble       (~10 минут, требует UniMorph rus.tsv)
3. RuT5               (~30-60 минут, требует UniMorph rus.tsv + GPU)
```

AnimacyClassifier и MetaEnsemble можно обучить на CPU за несколько минут. RuT5 требует GPU (RTX 5070 или аналог).

---

## Подготовка окружения

```bash
# Создайте виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# CPU-зависимости (обязательные)
pip install pymorphy3 pydantic fastapi "uvicorn[standard]" natasha

# GPU-зависимости
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers sentencepiece accelerate

# Navec embeddings (для AnimacyClassifier)
pip install navec

# Скачайте Navec-модель (~50 МБ)
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

# Скачайте UniMorph Russian (для MetaEnsemble и RuT5)
mkdir -p data
wget -O data/rus.tsv https://raw.githubusercontent.com/unimorph/rus/master/rus

# Создайте директории для моделей
mkdir -p models/animacy models/ensemble models/rut5-declension
```

---

## 1. AnimacyClassifier

### Что это

Бинарный классификатор, предсказывающий одушевлённость существительных. Критически важен для корректного винительного падежа OOV-слов (одушевлённые: Вин=Род, неодушевлённые: Вин=Им).

### Данные

Данные извлекаются автоматически из pymorphy3 (словарь OpenCorpora). Скрипт перебирает ~200 базовых слов + суффиксные вариации, получая парсинги с метками `anim`/`inan`. Типичный выход: 100–300 лемм. Для расширения можно добавить свой список слов в функцию `_get_word_list()` в `train_animacy.py`.

### Архитектура

```
Input:  [Navec embedding (300d)  |  suffix hash (30d)]  = 330d
         ↓
Linear(330, 128) → ReLU → Dropout(0.3)
         ↓
Linear(128, 64) → ReLU → Dropout(0.2)
         ↓
Linear(64, 1) → sigmoid → P(animate)
```

Суффиксные признаки: последние 1–5 символов слова хешируются в 30-мерный вектор. Это обеспечивает покрытие OOV-слов, даже если их нет в Navec-словаре, — суффикс `-тель` будет иметь похожий хеш для `учитель` и `обогатитель`.

### Обучение

```bash
python -m russian_declension.gpu.training.train_animacy \
    --output models/animacy \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.001 \
    --device auto          # auto/cuda/cpu

# Ожидаемый выход:
# Извлечено 180 лемм: 65 одушевлённых, 115 неодушевлённых
# После балансировки: 130 (50/50)
# ...
# Epoch 20/20  train_loss=0.2341  val_loss=0.3012  val_acc=0.8846  (0.1s)
# Test: loss=0.2987, accuracy=0.8571
# Готово. Артефакты: models/animacy/animacy_classifier.pt, config.json
```

### Артефакты

```
models/animacy/
├── animacy_classifier.pt   # PyTorch модель (~50 КБ)
└── config.json             # Гиперпараметры и метрики
```

### Подключение к API

```bash
ANIMACY_MODEL_PATH=models/animacy python -m russian_declension.main
```

### Расширение данных для лучшей точности

Для повышения точности с ~85% до ~92%+ рекомендуется расширить обучающую выборку:

```python
# Добавьте в _get_word_list() слова из вашего домена:
# Юридический: "истец", "ответчик", "свидетель", "потерпевший", ...
# Медицинский: "пациент", "хирург", "терапевт", "фармацевт", ...
# Технический: "программист", "сервер", "база", "протокол", ...
```

Также можно извлечь больше лемм из UniMorph:

```python
# В train_animacy.py, добавьте после extract_animacy_data():
with open("data/rus.tsv") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and "ANIM" in parts[2]:
            data.append((parts[0], 1))
        elif len(parts) >= 3 and "INAN" in parts[2]:
            data.append((parts[0], 0))
```

---

## 2. MetaEnsemble

### Что это

Мета-модель, которая учится выбирать лучший результат из нескольких engine'ов (pymorphy3, ruT5, char-level transformer). Вместо жёсткой fallback-цепочки, где каждый engine решает «уверен / передаю дальше», мета-модель получает все результаты одновременно и выбирает наилучший.

### Данные

Скрипт берёт UniMorph Russian (ground truth) и для каждой пары (лемма, целевая форма):
1. Прогоняет лемму через pymorphy3 → реальный результат + confidence
2. Симулирует результат ruT5 (P=0.95 правильный если pymorphy прав, P=0.70 если нет)
3. Симулирует результат char-transformer (P=0.92 / P=0.55)
4. Определяет label: какой engine дал правильный ответ

Симуляция нужна потому, что ruT5 и char-transformer могут быть ещё не обучены. После их обучения можно переобучить MetaEnsemble на реальных результатах для лучшей точности (см. «Переобучение на реальных данных» ниже).

### Архитектура

```
Input: 16 features
  Per engine (×3): [confidence, is_high_conf, len_ratio, changed]  = 12
  Suffix chars (×3): [ord(char) / 1200]                           = 3
  Word length: [len / 20]                                          = 1
         ↓
Linear(16, 32) → ReLU → Dropout(0.2)
         ↓
Linear(32, 32) → ReLU → Dropout(0.1)
         ↓
Linear(32, 3) → softmax → P(engine_i is correct)
```

### Обучение

```bash
python -m russian_declension.gpu.training.train_ensemble \
    --data data/rus.tsv \
    --output models/ensemble \
    --epochs 15 \
    --batch-size 512 \
    --max-samples 50000 \
    --device auto

# Ожидаемый выход:
# UniMorph: 58234 примеров
# Обрезано до 50000 примеров
# Генерация engine-результатов...
#   10000/50000 (3.2s)
#   ...
# Данные: 50000 примеров, 16 features. Распределение: {0: 45123, 1: 3456, 2: 1421}
# ...
# Epoch 15/15  train_loss=0.2810  val_loss=0.3142  val_acc=0.9234  (0.3s)
# Test: loss=0.3098, accuracy=0.9187
# Baseline (always pymorphy): 0.9025
# Прирост мета-модели: +1.62 п.п.
# Готово.
```

### Артефакты

```
models/ensemble/
├── meta_ensemble.pt   # PyTorch модель (~5 КБ)
└── config.json        # Гиперпараметры, baseline, прирост
```

### Подключение к API

```bash
ENSEMBLE_MODEL_PATH=models/ensemble python -m russian_declension.main
```

### Переобучение на реальных данных

После обучения ruT5 и char-level transformer рекомендуется переобучить MetaEnsemble на реальных (не симулированных) результатах:

```python
# Замените simulate_engine_results() в train_ensemble.py на:
from russian_declension.engines.pymorphy_engine import PymorphyEngine
from russian_declension.gpu.rut5_engine import RuT5Engine

pymorphy = PymorphyEngine()
rut5 = RuT5Engine(model_path="models/rut5-declension")

def real_engine_results(lemma, target_form, case_code, num_code):
    results = []
    # Реальный pymorphy
    r = pymorphy.inflect(lemma, Case(case_code), Number(num_code))
    results.append({"engine": "pymorphy", "result": r.inflected_form,
                    "confidence": r.confidence,
                    "correct": r.inflected_form.lower() == target_form.lower()})
    # Реальный ruT5
    r = rut5.inflect(lemma, Case(case_code), Number(num_code))
    results.append({"engine": "rut5", "result": r.inflected_form,
                    "confidence": r.confidence,
                    "correct": r.inflected_form.lower() == target_form.lower()})
    # ...
    return results
```

---

## 3. RuT5

### Что это

Fine-tuned ruT5-small для генерации словоформ OOV-слов. Базовая модель `cointegrated/rut5-small` уже «знает» русскую морфологию из предобучения — fine-tuning активирует эти знания для конкретной задачи инфлекции.

### Данные

UniMorph Russian (~60K троек). Формат: `лемма <TAB> форма <TAB> признаки`.

### Обучение

```bash
python -m russian_declension.gpu.training.train_rut5 \
    --data data/rus.tsv \
    --base-model cointegrated/rut5-small \
    --output models/rut5-declension \
    --epochs 10 \
    --batch-size 64 \
    --lr 3e-4

# Время: ~30-60 минут на RTX 5070
# VRAM: ~3-4 ГБ при batch_size=64
```

Для более мощной модели (если ruT5-small недостаточна):

```bash
python -m russian_declension.gpu.training.train_rut5 \
    --data data/rus.tsv \
    --base-model ai-forever/FRED-T5-large \
    --output models/fred-t5-declension \
    --epochs 5 \
    --batch-size 16 \
    --lr 1e-4
```

### Подключение к API

```bash
RUT5_MODEL_PATH=models/rut5-declension python -m russian_declension.main
```

---

## Запуск со всеми GPU-компонентами

```bash
RUT5_MODEL_PATH=models/rut5-declension \
ANIMACY_MODEL_PATH=models/animacy \
ENSEMBLE_MODEL_PATH=models/ensemble \
GPU_DEVICE=auto \
python -m russian_declension.main
```

Healthcheck (`GET /api/v1/health`) покажет статус каждого GPU-компонента:

```json
{
  "gpu_components": {
    "rut5": "active",
    "animacy_clf": "active",
    "ensemble": "active"
  }
}
```

---

## Бюджет VRAM на RTX 5070

| Компонент | При обучении | При инференсе |
|---|---|---|
| RuT5-small fine-tuning | ~3–4 ГБ | ~500 МБ |
| AnimacyClassifier training | <100 МБ | ~100 МБ |
| MetaEnsemble training | <50 МБ | <10 МБ |
| CUDA overhead | ~500 МБ | ~500 МБ |
| **Итого при инференсе** | — | **~1.1 ГБ** |

Остаётся ~11 ГБ свободного VRAM — достаточно для FRED-T5-large (ещё +1.5 ГБ) или будущих BERT-парсеров.

---

## Troubleshooting

**«Navec-модель не найдена»** → Скачайте файл `navec_hudlit_v1_12B_500K_300d_100q.tar` и положите рядом с проектом или в `models/animacy/`.

**«Слишком мало данных»** для AnimacyClassifier → Расширьте список слов в `_get_word_list()` или подключите UniMorph как дополнительный источник (см. выше).

**MetaEnsemble даёт маленький прирост** → Это ожидаемо при симулированных данных. Переобучите на реальных результатах engine'ов после обучения ruT5.

**ruT5 обучение падает с OOM** → Уменьшите `--batch-size` до 32 или 16.

**Baseline MetaEnsemble ~90%** → Это значит, что pymorphy3 прав в 90% случаев. Мета-модель полезна именно для оставшихся 10%, где она корректно переключается на ruT5 или char-transformer.
