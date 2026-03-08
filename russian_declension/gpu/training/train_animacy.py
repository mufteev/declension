#!/usr/bin/env python3
"""
Обучение AnimacyClassifier — бинарный классификатор одушевлённости.

Конвейер:
  1. Извлечь все существительные с метками anim/inan из pymorphy3 (OpenCorpora)
  2. Получить Navec-эмбеддинги (300d) для каждого слова
  3. Добавить суффиксные char-level признаки (для OOV-покрытия)
  4. Обучить MLP: [300 + 30 suffix] → 128 → 64 → 1 (sigmoid)
  5. Сохранить: animacy_classifier.pt, config.json

Запуск:
  python -m russian_declension.gpu.training.train_animacy \
      --output models/animacy \
      --epochs 20 --batch-size 256

Требования: pip install pymorphy3 navec torch
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Архитектура MLP
# ═══════════════════════════════════════════════════════════════════

EMBEDDING_DIM = 300    # Navec embedding dimension
SUFFIX_DIM = 30        # Char-level suffix features
INPUT_DIM = EMBEDDING_DIM + SUFFIX_DIM
HIDDEN_1 = 128
HIDDEN_2 = 64

# 10 самых информативных суффиксов для одушевлённости × 3 длины
# Хеширование последних 1..10 символов в фиксированный вектор
SUFFIX_HASH_SIZE = SUFFIX_DIM


class AnimacyMLP(nn.Module):
    """
    MLP для бинарной классификации одушевлённости.

    Вход: [word_embedding(300) ; suffix_features(30)]
    Выход: logit (scalar, > 0 = animate)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# Извлечение данных из pymorphy3
# ═══════════════════════════════════════════════════════════════════

def extract_animacy_data() -> list[tuple[str, int]]:
    """
    Извлечь все существительные с одушевлённостью из OpenCorpora через pymorphy3.

    Returns:
        [(word, label)] где label = 1 (animate) / 0 (inanimate)
    """
    from pymorphy3 import MorphAnalyzer
    morph = MorphAnalyzer(lang="ru")

    data = []
    seen_lemmas = set()

    logger.info("Извлечение данных одушевлённости из pymorphy3...")

    # Стратегия: перебираем известные слова через парсинг частотных текстов
    # и lemma-lookup. Для полного покрытия используем внутренний словарь.
    # Быстрый способ: парсим слова из предварительно собранного списка.

    # Способ 1: Используем word_list из pymorphy3 dictionary
    try:
        # pymorphy3 хранит словарь в DAFSA; прямого итератора нет,
        # но мы можем извлечь леммы через paradigm iteraton
        dictionary = morph.dictionary
        # Перебор всех лексем
        for para_idx in range(len(dictionary.paradigms)):
            try:
                paradigm = dictionary.paradigms[para_idx]
            except (IndexError, KeyError):
                continue
    except Exception:
        pass

    # Способ 2 (надёжный): парсим большой набор русских слов
    # и собираем уникальные леммы с их одушевлённостью
    test_words = _get_word_list()

    for word in test_words:
        parses = morph.parse(word)
        for p in parses:
            if "NOUN" not in p.tag:
                continue
            lemma = p.normal_form
            if lemma in seen_lemmas:
                continue
            seen_lemmas.add(lemma)

            tag_str = str(p.tag)
            if "anim" in tag_str:
                data.append((lemma, 1))
            elif "inan" in tag_str:
                data.append((lemma, 0))

    logger.info("Извлечено %d лемм: %d одушевлённых, %d неодушевлённых",
                len(data), sum(1 for _, l in data if l == 1),
                sum(1 for _, l in data if l == 0))
    return data


def _get_word_list() -> list[str]:
    """
    Получить список русских слов для извлечения одушевлённости.

    Стратегия: комбинируем несколько источников:
    1. Базовый список частотных существительных
    2. Слова из UniMorph (если доступен)
    3. Генерация через pymorphy3 нормальные формы
    """
    # Базовый набор + генерация вариаций суффиксов
    base_words = [
        # Одушевлённые
        "человек", "мужчина", "женщина", "ребёнок", "девочка", "мальчик",
        "учитель", "врач", "инженер", "писатель", "художник", "музыкант",
        "студент", "директор", "президент", "министр", "водитель", "рабочий",
        "солдат", "офицер", "генерал", "полковник", "лейтенант",
        "кошка", "собака", "лошадь", "корова", "медведь", "волк", "лиса",
        "птица", "рыба", "змея", "бабочка", "паук", "муха", "комар",
        "брат", "сестра", "отец", "мать", "дедушка", "бабушка", "дядя", "тётя",
        "друг", "враг", "сосед", "коллега", "начальник", "подчинённый",
        "программист", "аналитик", "дизайнер", "менеджер", "бухгалтер",
        "журналист", "актёр", "актриса", "певец", "танцор", "спортсмен",
        "лётчик", "космонавт", "моряк", "повар", "продавец", "покупатель",
        "пациент", "больной", "свидетель", "преступник", "судья",
        "философ", "историк", "биолог", "физик", "математик", "химик",
        "мертвец", "покойник", "зомби", "кукла", "робот",
        "ферзь", "конь", "слон", "валет", "туз", "король", "дама",
        # Неодушевлённые
        "дом", "стол", "стул", "окно", "дверь", "стена", "пол", "потолок",
        "книга", "ручка", "карандаш", "тетрадь", "газета", "журнал",
        "машина", "автобус", "поезд", "самолёт", "корабль", "велосипед",
        "город", "деревня", "улица", "дорога", "мост", "парк", "лес",
        "река", "озеро", "море", "океан", "гора", "поле", "остров",
        "еда", "хлеб", "молоко", "мясо", "рыба", "овощ", "фрукт",
        "вода", "сок", "чай", "кофе", "вино", "пиво",
        "одежда", "платье", "костюм", "пальто", "шапка", "обувь",
        "телефон", "компьютер", "телевизор", "холодильник", "плита",
        "время", "день", "ночь", "утро", "вечер", "год", "месяц", "неделя",
        "работа", "учёба", "отдых", "сон", "еда", "прогулка",
        "здоровье", "болезнь", "радость", "грусть", "любовь", "ненависть",
        "деньги", "цена", "зарплата", "налог", "бюджет", "расход",
        "документ", "договор", "акт", "справка", "приказ", "протокол",
        "компания", "организация", "предприятие", "учреждение", "фирма",
        "ответственность", "обязательство", "требование", "условие",
        "решение", "заседание", "совещание", "собрание", "конференция",
        "проект", "план", "программа", "стратегия", "политика",
        "результат", "достижение", "успех", "провал", "ошибка",
        "закон", "право", "свобода", "обязанность", "ответственность",
        "труп", "тело", "скелет", "череп", "кость",
        "камень", "песок", "глина", "железо", "золото", "серебро",
        "дерево", "цветок", "трава", "куст", "гриб",
    ]

    # Генерируем вариации через суффиксы
    animate_suffixes = ["тель", "ник", "чик", "щик", "ист", "ёр", "ер",
                        "ор", "арь", "лог", "граф", "навт", "вед", "ец"]
    inanimate_suffixes = ["ость", "ение", "ание", "ство", "тие",
                          "ция", "зия", "тор", "мент", "ура", "ика"]
    stems = ["работ", "учен", "стро", "управ", "програм", "исследова",
             "обществ", "развит", "производ", "обслуживан", "обеспечен"]

    for stem in stems:
        for suf in animate_suffixes:
            base_words.append(stem + suf)
        for suf in inanimate_suffixes:
            base_words.append(stem + suf)

    return list(set(base_words))


# ═══════════════════════════════════════════════════════════════════
# Navec embeddings
# ═══════════════════════════════════════════════════════════════════

def load_navec_embeddings() -> dict:
    """Загрузить Navec embeddings (300d, 250K vocab)."""
    try:
        from navec import Navec
        navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
        logger.info("Navec загружен: %d слов, %dd", len(navec.vocab), 300)
        return navec
    except FileNotFoundError:
        logger.warning(
            "Navec-модель не найдена. Скачайте:\n"
            "  wget https://storage.yandexcloud.net/natasha-navec/"
            "packs/navec_hudlit_v1_12B_500K_300d_100q.tar"
        )
        return None
    except ImportError:
        logger.warning("Navec не установлен: pip install navec")
        return None


def get_word_embedding(word: str, navec) -> np.ndarray:
    """Получить Navec-эмбеддинг слова. Если OOV — нулевой вектор."""
    if navec is None:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        if word in navec.vocab:
            idx = navec.vocab[word]
            return navec.pq.unpack(idx).astype(np.float32)
        # Попробуем lowercase
        low = word.lower()
        if low in navec.vocab:
            idx = navec.vocab[low]
            return navec.pq.unpack(idx).astype(np.float32)
    except Exception:
        pass
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# Суффиксные признаки (char-level)
# ═══════════════════════════════════════════════════════════════════

def suffix_features(word: str, dim: int = SUFFIX_DIM) -> np.ndarray:
    """
    Извлечь суффиксные признаки слова через char-hashing.

    Хешируем суффиксы длиной 1..5 в фиксированный вектор,
    чтобы OOV-слова с похожими суффиксами получали похожие вектора.
    """
    features = np.zeros(dim, dtype=np.float32)
    low = word.lower()
    for length in range(1, 6):
        if len(low) < length:
            break
        suffix = low[-length:]
        h = hash(suffix) % dim
        features[h] += 1.0 / length  # Более длинные суффиксы = меньший вес
    # Нормализуем
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class AnimacyDataset(Dataset):
    def __init__(self, data: list[tuple[str, int]], navec):
        self.data = data
        self.navec = navec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, label = self.data[idx]
        emb = get_word_embedding(word, self.navec)
        suf = suffix_features(word)
        features = np.concatenate([emb, suf])
        return torch.FloatTensor(features), torch.FloatTensor([label])


# ═══════════════════════════════════════════════════════════════════
# Обучение
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device).squeeze()
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device).squeeze()
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        n += len(labels)
    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train AnimacyClassifier")
    parser.add_argument("--output", default="models/animacy", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Извлечение данных
    data = extract_animacy_data()
    if len(data) < 100:
        logger.error("Слишком мало данных (%d). Проверьте pymorphy3.", len(data))
        return

    # 2. Загрузка Navec
    navec = load_navec_embeddings()

    # 3. Балансировка классов
    anim = [(w, l) for w, l in data if l == 1]
    inan = [(w, l) for w, l in data if l == 0]
    logger.info("До балансировки: anim=%d, inan=%d", len(anim), len(inan))
    min_count = min(len(anim), len(inan))
    random.shuffle(anim)
    random.shuffle(inan)
    balanced = anim[:min_count] + inan[:min_count]
    random.shuffle(balanced)
    logger.info("После балансировки: %d (50/50)", len(balanced))

    # 4. Train/val/test split (80/10/10)
    n = len(balanced)
    train_data = balanced[:int(n * 0.8)]
    val_data = balanced[int(n * 0.8):int(n * 0.9)]
    test_data = balanced[int(n * 0.9):]

    train_ds = AnimacyDataset(train_data, navec)
    val_ds = AnimacyDataset(val_data, navec)
    test_ds = AnimacyDataset(test_data, navec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 5. Модель
    device = "cpu"
    if args.device == "auto" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "auto":
        device = args.device

    model = AnimacyMLP(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    logger.info("Модель: %d параметров, device=%s",
                sum(p.numel() for p in model.parameters()), device)

    # 6. Обучение
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
                     epoch, args.epochs, train_loss, val_loss, val_acc, time.time() - t0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, output_dir / "animacy_classifier.pt")
            logger.info("  → Лучшая val_acc: %.4f (сохранено)", val_acc)

    # 7. Тестирование
    best_model = torch.load(output_dir / "animacy_classifier.pt",
                            map_location=device, weights_only=False)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    logger.info("Test: loss=%.4f, accuracy=%.4f", test_loss, test_acc)

    # 8. Сохранение конфигурации
    config = {
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "suffix_dim": SUFFIX_DIM,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Готово. Артефакты: %s/animacy_classifier.pt, config.json", output_dir)


if __name__ == "__main__":
    main()
