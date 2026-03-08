#!/usr/bin/env python3
"""
Обучение MetaEnsemble — мета-модель выбора лучшего engine'а.

Идея (Kuzmenko 2016, HSE): engine'ы ошибаются на РАЗНЫХ словах.
Мета-модель учится предсказывать, какой engine прав для конкретного
входа, на основе их confidence-скоров и признаков слова.

Конвейер:
  1. Загрузить UniMorph Russian (ground truth)
  2. Прогнать каждое слово через pymorphy3 → получить result + confidence
  3. Симулировать результаты ruT5 / char-transformer (с шумовой моделью)
  4. Для каждого примера: [engine_features + word_features] → label (кто прав)
  5. Обучить MLP: 16 features → 32 → 3 (softmax, 3 engine'а)
  6. Сохранить: meta_ensemble.pt, config.json

Запуск:
  python -m russian_declension.gpu.training.train_ensemble \
      --data data/rus.tsv \
      --output models/ensemble \
      --epochs 15 --batch-size 512

Требования: pip install pymorphy3 torch
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════════

NUM_ENGINES = 3           # pymorphy, rut5, neural
FEATURES_PER_ENGINE = 4   # confidence, is_high_conf, len_ratio, changed
SUFFIX_FEATURES = 3       # char codes последних 3 символов
WORD_FEATURES = 1         # длина
TOTAL_FEATURES = NUM_ENGINES * FEATURES_PER_ENGINE + SUFFIX_FEATURES + WORD_FEATURES
# = 3*4 + 3 + 1 = 16

HIDDEN_DIM = 32

UNIMORPH_CASE = {"NOM":"nomn","GEN":"gent","DAT":"datv",
                  "ACC":"accs","INS":"ablt","ESS":"loct","LOC":"loct"}
UNIMORPH_NUM = {"SG":"sing","PL":"plur"}


class EnsembleMLP(nn.Module):
    """MLP для 3-way classification: какой engine прав."""

    def __init__(self, input_dim: int = TOTAL_FEATURES, num_engines: int = NUM_ENGINES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, num_engines),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# Генерация обучающих данных
# ═══════════════════════════════════════════════════════════════════

def load_unimorph(path: str) -> list[tuple[str, str, str, str]]:
    """
    Загрузить UniMorph → (lemma, target_form, case_code, number_code).
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            lemma, form, feats = row[0], row[1], row[2]
            tags = feats.split(";")
            case_code = next((UNIMORPH_CASE[t] for t in tags if t in UNIMORPH_CASE), None)
            num_code = next((UNIMORPH_NUM[t] for t in tags if t in UNIMORPH_NUM), None)
            if case_code:
                data.append((lemma, form, case_code, num_code or "sing"))
    return data


def simulate_engine_results(
    lemma: str, target_form: str, case_code: str, num_code: str, morph
) -> list[dict]:
    """
    Симулировать результаты от трёх engine'ов для одной пары (lemma, target).

    Returns:
        [
          {"engine": "pymorphy", "result": "...", "confidence": 0.9, "correct": True/False},
          {"engine": "rut5", "result": "...", "confidence": 0.8, "correct": True/False},
          {"engine": "neural", "result": "...", "confidence": 0.7, "correct": True/False},
        ]
    """
    results = []

    # ── Engine 1: pymorphy3 (реальный результат) ─────────────────
    pymorphy_result = None
    pymorphy_conf = 0.0
    parses = morph.parse(lemma)
    noun_parses = [p for p in parses if "NOUN" in p.tag]
    if noun_parses:
        best = max(noun_parses, key=lambda p: p.score)
        target_grammemes = frozenset({case_code, num_code})
        inflected = best.inflect(target_grammemes)
        if inflected:
            pymorphy_result = inflected.word
            # Реалистичная confidence
            methods = best.methods_stack
            is_dict = methods and "Dictionary" in methods[0][0].__class__.__name__
            pymorphy_conf = max(best.score, 0.85) if is_dict else min(best.score * 0.6, 0.6)

    if pymorphy_result is None:
        pymorphy_result = lemma
        pymorphy_conf = 0.1

    results.append({
        "engine": "pymorphy",
        "result": pymorphy_result,
        "confidence": pymorphy_conf,
        "correct": pymorphy_result.lower() == target_form.lower(),
    })

    # ── Engine 2: simulated ruT5 ─────────────────────────────────
    # Модель ruT5 после fine-tuning на UniMorph показывает ~94-97% accuracy.
    # Для OOV-слов (где pymorphy слаб) ruT5 сильнее.
    # Симулируем: если pymorphy правильный → ruT5 тоже с P=0.95;
    #              если pymorphy неправильный → ruT5 правильный с P=0.7.
    if results[0]["correct"]:
        rut5_correct = random.random() < 0.95
    else:
        rut5_correct = random.random() < 0.70

    if rut5_correct:
        rut5_result = target_form
        rut5_conf = random.uniform(0.65, 0.95)
    else:
        # Генерируем ошибочную форму (лёгкая пертурбация)
        rut5_result = _perturb_word(target_form)
        rut5_conf = random.uniform(0.3, 0.7)

    results.append({
        "engine": "rut5",
        "result": rut5_result,
        "confidence": rut5_conf,
        "correct": rut5_correct,
    })

    # ── Engine 3: simulated char-level transformer ───────────────
    # Char-level ~91-96% accuracy, слабее ruT5 на OOV но быстрее.
    if results[0]["correct"]:
        neural_correct = random.random() < 0.92
    else:
        neural_correct = random.random() < 0.55

    if neural_correct:
        neural_result = target_form
        neural_conf = random.uniform(0.5, 0.9)
    else:
        neural_result = _perturb_word(target_form)
        neural_conf = random.uniform(0.2, 0.6)

    results.append({
        "engine": "neural",
        "result": neural_result,
        "confidence": neural_conf,
        "correct": neural_correct,
    })

    return results


def _perturb_word(word: str) -> str:
    """Создать ошибочную словоформу через замену/удаление/добавление символа."""
    if len(word) < 2:
        return word
    chars = list(word)
    op = random.choice(["replace", "delete", "insert"])
    pos = random.randint(max(0, len(chars) - 3), len(chars) - 1)

    russian_vowels = "аеёиоуыэюя"
    russian_consonants = "бвгджзклмнпрстфхцчшщ"

    if op == "replace":
        chars[pos] = random.choice(russian_vowels + russian_consonants)
    elif op == "delete" and len(chars) > 2:
        chars.pop(pos)
    elif op == "insert":
        chars.insert(pos, random.choice(russian_vowels))

    return "".join(chars)


# ═══════════════════════════════════════════════════════════════════
# Feature extraction (должен совпадать с runtime!)
# ═══════════════════════════════════════════════════════════════════

def extract_features(engine_results: list[dict], word: str) -> np.ndarray:
    """
    Извлечь вектор признаков — ДОЛЖЕН совпадать с MetaEnsemble._extract_features().

    Порядок: [pymorphy_4_features, rut5_4_features, neural_4_features,
              suffix_3_features, word_length_1]
    Итого: 16 features.
    """
    features = []
    engine_order = ["pymorphy", "rut5", "neural"]
    by_engine = {r["engine"]: r for r in engine_results}

    for eng in engine_order:
        r = by_engine.get(eng)
        if r:
            features.extend([
                r["confidence"],
                1.0 if r["confidence"] > 0.8 else 0.0,
                len(r["result"]) / max(len(word), 1),
                1.0 if r["result"] != word else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    # Суффиксные признаки
    for i in range(1, 4):
        if len(word) >= i:
            features.append(ord(word[-i]) / 1200.0)
        else:
            features.append(0.0)

    # Длина
    features.append(len(word) / 20.0)

    return np.array(features, dtype=np.float32)


def determine_label(engine_results: list[dict]) -> int:
    """
    Определить label: индекс engine'а, который дал правильный ответ.

    Приоритет (при нескольких правильных): pymorphy > rut5 > neural.
    Если никто не прав: label = 0 (pymorphy как fallback).
    """
    engine_order = ["pymorphy", "rut5", "neural"]
    by_engine = {r["engine"]: r for r in engine_results}

    for idx, eng in enumerate(engine_order):
        r = by_engine.get(eng)
        if r and r["correct"]:
            return idx
    return 0  # fallback to pymorphy


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class EnsembleDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════
# Обучение
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
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
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        n += len(labels)
    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MetaEnsemble")
    parser.add_argument("--data", required=True, help="UniMorph rus.tsv")
    parser.add_argument("--output", default="models/ensemble")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Макс. примеров (для скорости)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка UniMorph
    raw = load_unimorph(args.data)
    logger.info("UniMorph: %d примеров", len(raw))
    if len(raw) > args.max_samples:
        random.shuffle(raw)
        raw = raw[:args.max_samples]
        logger.info("Обрезано до %d примеров", len(raw))

    # 2. Инициализация pymorphy3
    from pymorphy3 import MorphAnalyzer
    morph = MorphAnalyzer(lang="ru")

    # 3. Генерация данных
    logger.info("Генерация engine-результатов...")
    all_features = []
    all_labels = []
    t0 = time.time()

    for i, (lemma, form, case_code, num_code) in enumerate(raw):
        engine_results = simulate_engine_results(lemma, form, case_code, num_code, morph)
        feat = extract_features(engine_results, lemma)
        label = determine_label(engine_results)
        all_features.append(feat)
        all_labels.append(label)

        if (i + 1) % 10000 == 0:
            logger.info("  %d/%d (%.1fs)", i + 1, len(raw), time.time() - t0)

    X = np.array(all_features)
    y = np.array(all_labels)
    logger.info("Данные: %d примеров, %d features. Распределение: %s",
                len(y), X.shape[1],
                dict(zip(*np.unique(y, return_counts=True))))

    # 4. Split
    n = len(y)
    idx = np.random.permutation(n)
    train_idx = idx[:int(n * 0.8)]
    val_idx = idx[int(n * 0.8):int(n * 0.9)]
    test_idx = idx[int(n * 0.9):]

    train_ds = EnsembleDataset(X[train_idx], y[train_idx])
    val_ds = EnsembleDataset(X[val_idx], y[val_idx])
    test_ds = EnsembleDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 5. Модель
    device = "cpu"
    if args.device == "auto" and torch.cuda.is_available():
        device = "cuda"
    elif args.device != "auto":
        device = args.device

    model = EnsembleMLP(TOTAL_FEATURES, NUM_ENGINES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("Модель: %d параметров, device=%s",
                sum(p.numel() for p in model.parameters()), device)

    # 6. Обучение
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0_ep = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  (%.1fs)",
                     epoch, args.epochs, train_loss, val_loss, val_acc,
                     time.time() - t0_ep)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, output_dir / "meta_ensemble.pt")
            logger.info("  → Лучшая val_acc: %.4f", val_acc)

    # 7. Тестирование
    best_model = torch.load(output_dir / "meta_ensemble.pt",
                            map_location=device, weights_only=False)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
    logger.info("Test: loss=%.4f, accuracy=%.4f", test_loss, test_acc)

    # Baseline: всегда выбирать pymorphy
    baseline_acc = (y[test_idx] == 0).mean()
    logger.info("Baseline (always pymorphy): %.4f", baseline_acc)
    logger.info("Прирост мета-модели: +%.2f п.п.", (test_acc - baseline_acc) * 100)

    # 8. Сохранение
    config = {
        "input_dim": TOTAL_FEATURES,
        "num_engines": NUM_ENGINES,
        "hidden_dim": HIDDEN_DIM,
        "engine_order": ["pymorphy", "rut5", "neural"],
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "baseline_acc": float(baseline_acc),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Готово. Артефакты: %s/meta_ensemble.pt, config.json", output_dir)


if __name__ == "__main__":
    main()
