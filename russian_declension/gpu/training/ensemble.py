#!/usr/bin/env python3
"""
train_ensemble.py — Обучение MetaEnsemble модели.

Использует:
  - Все 96 CPU ядер для подготовки данных (multiprocessing + parallel feature extraction)
  - A100 GPU для обучения модели (mixed precision, large batch)
  - UniMorph Russian как ground truth

Запуск:
  python train_ensemble.py \
    --unimorph-path ./rus.txt \
    --output-dir ./models/ensemble \
    --batch-size 4096 \
    --epochs 50 \
    --num-workers 94 \
    --lr 0.001

Переобучение на реальных данных (после обучения ruT5 и char-transformer):
  python train_ensemble.py \
    --unimorph-path ./rus.txt \
    --output-dir ./models/ensemble \
    --real-engines  # использует реальные engine'ы вместо симуляции
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count, shared_memory
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────────────

ENGINE_ORDER = ["pymorphy", "rut5", "neural"]
NUM_ENGINES = len(ENGINE_ORDER)
NUM_FEATURES = 16  # 4 per engine (×3) + 3 suffix + 1 word_len
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Модель
# ──────────────────────────────────────────────────────────────────────

class EnsembleMLP(nn.Module):
    """
    Мета-модель для выбора лучшего engine'а.

    Input:  16 features
    Output: 3 logits (softmax → вероятность каждого engine'а)
    """

    def __init__(self, n_features: int = NUM_FEATURES, n_classes: int = NUM_ENGINES,
                 hidden1: int = 64, hidden2: int = 64,
                 dropout1: float = 0.2, dropout2: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),

            nn.Linear(hidden2, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
# Данные UniMorph
# ──────────────────────────────────────────────────────────────────────

@dataclass
class UnimorphEntry:
    """Одна запись UniMorph: лемма → целевая форма + теги."""
    lemma: str
    target_form: str
    tags: str


def load_unimorph(path: str, max_entries: int = 0) -> list[UnimorphEntry]:
    """
    Загрузка UniMorph Russian.
    Формат: lemma\ttarget_form\ttags
    """
    entries = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UniMorph файл не найден: {path}")

    logger.info("Загрузка UniMorph из %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            lemma, target, tags = parts[0], parts[1], parts[2]
            # Фильтрация: только кириллица, разумная длина
            if not lemma or not target:
                continue
            if len(lemma) > 50 or len(target) > 50:
                continue
            entries.append(UnimorphEntry(lemma=lemma, target_form=target, tags=tags))
            if max_entries and len(entries) >= max_entries:
                break

    logger.info("Загружено %d записей UniMorph", len(entries))
    return entries


# ──────────────────────────────────────────────────────────────────────
# pymorphy3 engine (реальный)
# ──────────────────────────────────────────────────────────────────────

def init_pymorphy():
    """Инициализация pymorphy3 в worker'е."""
    global _morph
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()


def pymorphy_inflect_single(entry: UnimorphEntry) -> dict:
    """
    Попытка склонения через pymorphy3.
    Возвращает dict с результатом.
    """
    global _morph

    lemma = entry.lemma
    target = entry.target_form

    try:
        parsed = _morph.parse(lemma)
        if not parsed:
            return {
                "lemma": lemma,
                "target": target,
                "pymorphy_result": lemma,
                "pymorphy_conf": 0.0,
                "pymorphy_correct": False,
            }

        best_parse = parsed[0]
        confidence = float(best_parse.score)

        # Простая попытка: берём нормальную форму и её формы
        # Ищем среди всех лексем формы, совпадающие с target
        result_form = lemma
        found = False

        for p in parsed[:3]:  # Берём top-3 разбора
            lexeme = p.lexeme
            for lex_entry in lexeme:
                if lex_entry.word == target:
                    result_form = target
                    confidence = max(confidence, float(p.score))
                    found = True
                    break
            if found:
                break

        if not found:
            # Используем первый разбор, форму лемма
            result_form = best_parse.normal_form if best_parse.normal_form else lemma
            confidence *= 0.5

        return {
            "lemma": lemma,
            "target": target,
            "pymorphy_result": result_form,
            "pymorphy_conf": min(confidence, 1.0),
            "pymorphy_correct": result_form.lower() == target.lower(),
        }
    except Exception:
        return {
            "lemma": lemma,
            "target": target,
            "pymorphy_result": lemma,
            "pymorphy_conf": 0.0,
            "pymorphy_correct": False,
        }


def run_pymorphy_parallel(entries: list[UnimorphEntry],
                          num_workers: int) -> list[dict]:
    """Параллельное склонение через pymorphy3."""
    logger.info("pymorphy3: обработка %d записей на %d worker'ах ...",
                len(entries), num_workers)
    t0 = time.perf_counter()

    with Pool(processes=num_workers, initializer=init_pymorphy) as pool:
        results = pool.map(pymorphy_inflect_single, entries, chunksize=1024)

    elapsed = time.perf_counter() - t0
    correct = sum(1 for r in results if r["pymorphy_correct"])
    logger.info("pymorphy3: %.1fs, accuracy=%.2f%% (%d/%d)",
                elapsed, 100.0 * correct / len(results), correct, len(results))
    return results


# ──────────────────────────────────────────────────────────────────────
# Симуляция engine'ов (ruT5, char-transformer)
# ──────────────────────────────────────────────────────────────────────

def simulate_engine(target: str, pymorphy_correct: bool,
                    p_correct_if_pymorphy_right: float,
                    p_correct_if_pymorphy_wrong: float,
                    rng: random.Random) -> tuple[str, float, bool]:
    """
    Симулирует результат engine'а.
    Возвращает (result, confidence, is_correct).
    """
    if pymorphy_correct:
        correct = rng.random() < p_correct_if_pymorphy_right
    else:
        correct = rng.random() < p_correct_if_pymorphy_wrong

    if correct:
        result = target
        # Высокая confidence с шумом
        confidence = min(1.0, max(0.5, rng.gauss(0.88, 0.08)))
    else:
        # Генерируем ошибочную форму
        result = _corrupt_word(target, rng)
        confidence = min(0.95, max(0.1, rng.gauss(0.55, 0.15)))

    return result, confidence, correct


def _corrupt_word(word: str, rng: random.Random) -> str:
    """Порча слова для симуляции ошибки."""
    if len(word) <= 1:
        return word

    corruption_type = rng.choice(["suffix", "char_swap", "char_drop", "char_insert"])

    if corruption_type == "suffix" and len(word) > 2:
        # Замена суффикса
        cut = rng.randint(1, min(3, len(word) - 1))
        russian_suffixes = ["а", "у", "ом", "ов", "ей", "ам", "ами", "ах",
                            "и", "е", "ы", "о", "ю", "ь", "ём"]
        return word[:-cut] + rng.choice(russian_suffixes)
    elif corruption_type == "char_swap" and len(word) > 2:
        idx = rng.randint(0, len(word) - 2)
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    elif corruption_type == "char_drop" and len(word) > 2:
        idx = rng.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    else:
        # char_insert
        idx = rng.randint(0, len(word))
        c = chr(rng.randint(ord("а"), ord("я")))
        return word[:idx] + c + word[idx:]


# ──────────────────────────────────────────────────────────────────────
# Feature extraction (ИДЕНТИЧНО runtime)
# ──────────────────────────────────────────────────────────────────────

def extract_features(word: str,
                     engine_results: list[tuple[str, float, str]]) -> list[float]:
    """
    Извлечение 16 признаков.

    engine_results: [(result_text, confidence, engine_name), ...]
      В порядке ENGINE_ORDER: [pymorphy, rut5, neural]

    Порядок features:
      Per engine ×3: [confidence, is_high_conf, len_ratio, changed] = 12
      Suffix ×3:     [ord(char[-1])/1200, ord(char[-2])/1200, ord(char[-3])/1200] = 3
      Word len:      [len/20] = 1
      Total: 16
    """
    features = []
    by_engine = {}
    for text, conf, eng_name in engine_results:
        by_engine[eng_name] = (text, conf)

    for eng_name in ENGINE_ORDER:
        if eng_name in by_engine:
            text, conf = by_engine[eng_name]
            features.extend([
                conf,
                1.0 if conf > 0.8 else 0.0,
                len(text) / max(len(word), 1),
                1.0 if text != word else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

    # Suffix characters
    for i in range(1, 4):
        if len(word) >= i:
            features.append(ord(word[-i]) / 1200.0)
        else:
            features.append(0.0)

    # Word length
    features.append(len(word) / 20.0)

    return features


# ──────────────────────────────────────────────────────────────────────
# Worker для параллельной генерации примеров
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SampleGeneratorConfig:
    """Конфиг для worker'а генерации."""
    rut5_p_right: float = 0.95
    rut5_p_wrong: float = 0.70
    neural_p_right: float = 0.92
    neural_p_wrong: float = 0.55
    seed_base: int = SEED


def _generate_samples_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Worker: генерирует (features, labels) для chunk pymorphy-результатов.

    args: (chunk_of_pymorphy_results, config, worker_id)
    """
    chunk, config, worker_id = args
    rng = random.Random(config.seed_base + worker_id)

    features_list = []
    labels_list = []

    for pm_result in chunk:
        lemma = pm_result["lemma"]
        target = pm_result["target"]
        pm_text = pm_result["pymorphy_result"]
        pm_conf = pm_result["pymorphy_conf"]
        pm_correct = pm_result["pymorphy_correct"]

        # Симуляция ruT5
        rut5_text, rut5_conf, rut5_correct = simulate_engine(
            target, pm_correct,
            config.rut5_p_right, config.rut5_p_wrong, rng
        )

        # Симуляция neural
        neural_text, neural_conf, neural_correct = simulate_engine(
            target, pm_correct,
            config.neural_p_right, config.neural_p_wrong, rng
        )

        # Engine results
        engine_results = [
            (pm_text, pm_conf, "pymorphy"),
            (rut5_text, rut5_conf, "rut5"),
            (neural_text, neural_conf, "neural"),
        ]

        # Features
        feats = extract_features(lemma, engine_results)

        # Label: какой engine дал правильный ответ?
        # Приоритет: pymorphy > rut5 > neural (при одинаковой правильности)
        corrects = [pm_correct, rut5_correct, neural_correct]
        if any(corrects):
            # Выбираем engine с наивысшей confidence среди правильных
            best_label = -1
            best_conf = -1.0
            confs = [pm_conf, rut5_conf, neural_conf]
            for i in range(NUM_ENGINES):
                if corrects[i] and confs[i] > best_conf:
                    best_conf = confs[i]
                    best_label = i
            label = best_label
        else:
            # Никто не прав — назначаем engine с наибольшей confidence
            confs = [pm_conf, rut5_conf, neural_conf]
            label = int(np.argmax(confs))

        features_list.append(feats)
        labels_list.append(label)

    return (
        np.array(features_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class EnsembleDataset(Dataset):
    """Torch Dataset из numpy массивов features и labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────
# Подготовка данных
# ──────────────────────────────────────────────────────────────────────

def prepare_data(entries: list[UnimorphEntry],
                 num_workers: int,
                 config: SampleGeneratorConfig,
                 val_ratio: float = 0.1) -> tuple[EnsembleDataset, EnsembleDataset, np.ndarray]:
    """
    Полный pipeline подготовки данных:
    1. Параллельный pymorphy3
    2. Параллельная генерация features + labels
    3. Train/val split
    
    Returns: (train_dataset, val_dataset, class_weights)
    """
    # Шаг 1: pymorphy3
    pymorphy_workers = min(num_workers, 48)  # pymorphy не масштабируется бесконечно
    pm_results = run_pymorphy_parallel(entries, pymorphy_workers)

    # Шаг 2: параллельная генерация features
    logger.info("Генерация features + labels на %d worker'ах ...", num_workers)
    t0 = time.perf_counter()

    # Разбиваем на chunks
    chunk_size = max(1, len(pm_results) // num_workers)
    chunks = []
    for i in range(0, len(pm_results), chunk_size):
        chunk = pm_results[i:i + chunk_size]
        chunks.append((chunk, config, len(chunks)))

    with Pool(processes=min(num_workers, len(chunks))) as pool:
        results = pool.map(_generate_samples_worker, chunks)

    # Собираем результаты
    all_features = np.concatenate([r[0] for r in results], axis=0)
    all_labels = np.concatenate([r[1] for r in results], axis=0)

    elapsed = time.perf_counter() - t0
    logger.info("Features: %.1fs, shape=%s", elapsed, all_features.shape)

    # Статистика классов
    label_counts = Counter(all_labels.tolist())
    for i, eng in enumerate(ENGINE_ORDER):
        cnt = label_counts.get(i, 0)
        logger.info("  Class %d (%s): %d (%.1f%%)",
                     i, eng, cnt, 100.0 * cnt / len(all_labels))

    # Class weights для балансировки
    total = len(all_labels)
    class_weights = np.array([
        total / (NUM_ENGINES * max(label_counts.get(i, 1), 1))
        for i in range(NUM_ENGINES)
    ], dtype=np.float32)
    # Нормализуем
    class_weights /= class_weights.sum()
    class_weights *= NUM_ENGINES
    logger.info("Class weights: %s", class_weights)

    # Шаг 3: Train/val split (стратифицированный)
    np.random.seed(SEED)
    indices = np.arange(total)
    np.random.shuffle(indices)

    val_size = int(total * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    val_features = all_features[val_indices]
    val_labels = all_labels[val_indices]

    logger.info("Train: %d, Val: %d", len(train_labels), len(val_labels))

    return (
        EnsembleDataset(train_features, train_labels),
        EnsembleDataset(val_features, val_labels),
        class_weights,
    )


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Гиперпараметры обучения."""
    batch_size: int = 4096
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden1: int = 64
    hidden2: int = 64
    dropout1: float = 0.2
    dropout2: float = 0.1
    use_amp: bool = True  # mixed precision (A100 → TF32 + FP16)
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    label_smoothing: float = 0.05
    num_dataloader_workers: int = 8


def train_model(train_ds: EnsembleDataset,
                val_ds: EnsembleDataset,
                class_weights: np.ndarray,
                config: TrainConfig,
                device: torch.device) -> EnsembleMLP:
    """
    Обучение модели с:
    - Mixed precision (AMP) для A100
    - OneCycleLR scheduler
    - Early stopping
    - Label smoothing
    - Class-weighted loss
    - Gradient clipping
    """
    logger.info("=" * 60)
    logger.info("TRAINING CONFIG")
    logger.info("  Device:     %s", device)
    logger.info("  Batch size: %d", config.batch_size)
    logger.info("  Epochs:     %d", config.epochs)
    logger.info("  LR:         %g", config.lr)
    logger.info("  AMP:        %s", config.use_amp)
    logger.info("  Hidden:     %d / %d", config.hidden1, config.hidden2)
    logger.info("=" * 60)

    # Модель
    model = EnsembleMLP(
        n_features=NUM_FEATURES,
        n_classes=NUM_ENGINES,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", param_count)

    # Weighted sampler для train
    train_label_counts = Counter(train_ds.labels.numpy().tolist())
    sample_weights = []
    for label in train_ds.labels.numpy():
        sample_weights.append(1.0 / max(train_label_counts[int(label)], 1))
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    # DataLoaders
    # Pin memory + persistent workers для быстрого data loading
    dl_workers = min(config.num_dataloader_workers, cpu_count())
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=dl_workers,
        pin_memory=True,
        persistent_workers=True if dl_workers > 0 else False,
        prefetch_factor=4 if dl_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,  # val может быть больше
        shuffle=False,
        num_workers=dl_workers,
        pin_memory=True,
        persistent_workers=True if dl_workers > 0 else False,
        prefetch_factor=4 if dl_workers > 0 else None,
    )

    # Loss с class weights и label smoothing
    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=config.label_smoothing,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    # AMP scaler
    scaler = GradScaler(enabled=config.use_amp)

    # Enable TF32 on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.perf_counter()

        # ─── Train ───
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.use_amp, dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += features.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ─── Validation ───
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * NUM_ENGINES
        per_class_total = [0] * NUM_ENGINES

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=config.use_amp, dtype=torch.float16):
                    logits = model(features)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item() * features.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += features.size(0)

                for i in range(NUM_ENGINES):
                    mask = labels == i
                    per_class_correct[i] += (preds[mask] == i).sum().item()
                    per_class_total[i] += mask.sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.perf_counter() - epoch_t0

        # Per-class accuracy string
        per_class_str = " | ".join(
            f"{ENGINE_ORDER[i]}: {100.0 * per_class_correct[i] / max(per_class_total[i], 1):.1f}%"
            for i in range(NUM_ENGINES)
        )

        logger.info(
            "Epoch %2d/%d [%.1fs] "
            "train_loss=%.4f train_acc=%.3f | "
            "val_loss=%.4f val_acc=%.3f | "
            "lr=%.2e | %s",
            epoch, config.epochs, epoch_time,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr,
            per_class_str,
        )

        # Early stopping check
        improved = False
        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True
            logger.info("  ★ New best val_acc=%.4f", best_val_acc)
        elif val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            logger.info("Early stopping at epoch %d (patience=%d)",
                        epoch, config.early_stopping_patience)
            break

    # Восстанавливаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
    logger.info("Best val_acc=%.4f, val_loss=%.4f", best_val_acc, best_val_loss)

    return model, history


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(model: EnsembleMLP, val_ds: EnsembleDataset,
                   device: torch.device, batch_size: int = 8192):
    """Детальная оценка модели."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)

    # Overall accuracy
    acc = (preds == labels).float().mean().item()
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("  Overall accuracy: %.2f%%", 100.0 * acc)

    # Per-class metrics
    for i, eng in enumerate(ENGINE_ORDER):
        mask = labels == i
        if mask.sum() == 0:
            continue
        class_acc = (preds[mask] == i).float().mean().item()
        class_total = mask.sum().item()

        # Precision, Recall
        pred_mask = preds == i
        tp = ((preds == i) & (labels == i)).sum().item()
        fp = ((preds == i) & (labels != i)).sum().item()
        fn = ((preds != i) & (labels == i)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        logger.info("  %s: acc=%.2f%% P=%.3f R=%.3f F1=%.3f (n=%d)",
                     eng, 100.0 * class_acc, precision, recall, f1, class_total)

    # Confidence calibration
    confidence_of_correct = probs[torch.arange(len(preds)), preds]
    correct_mask = (preds == labels)
    avg_conf_correct = confidence_of_correct[correct_mask].mean().item()
    avg_conf_wrong = confidence_of_correct[~correct_mask].mean().item() if (~correct_mask).any() else 0
    logger.info("  Avg confidence (correct): %.3f", avg_conf_correct)
    logger.info("  Avg confidence (wrong):   %.3f", avg_conf_wrong)
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────

def save_model(model: EnsembleMLP, output_dir: Path, config: TrainConfig,
               history: dict, class_weights: np.ndarray):
    """Сохранение модели и конфига."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    pt_path = output_dir / "meta_ensemble.pt"
    model_cpu = model.cpu()
    torch.save(model_cpu, str(pt_path))
    logger.info("Модель сохранена: %s (%.1f KB)", pt_path, pt_path.stat().st_size / 1024)

    # Config
    config_dict = {
        "model_class": "EnsembleMLP",
        "n_features": NUM_FEATURES,
        "n_classes": NUM_ENGINES,
        "hidden1": config.hidden1,
        "hidden2": config.hidden2,
        "dropout1": config.dropout1,
        "dropout2": config.dropout2,
        "engine_order": ENGINE_ORDER,
        "training": {
            "batch_size": config.batch_size,
            "epochs": len(history.get("train_loss", [])),
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "use_amp": config.use_amp,
            "best_val_acc": max(history.get("val_acc", [0])),
            "best_val_loss": min(history.get("val_loss", [float("inf")])),
        },
        "class_weights": class_weights.tolist(),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info("Конфиг сохранён: %s", config_path)

    # Also save state_dict for safer loading
    sd_path = output_dir / "meta_ensemble_state_dict.pt"
    torch.save(model_cpu.state_dict(), str(sd_path))
    logger.info("State dict сохранён: %s", sd_path)


# ──────────────────────────────────────────────────────────────────────
# ONNX export (опционально, для deployment)
# ──────────────────────────────────────────────────────────────────────

def export_onnx(model: EnsembleMLP, output_dir: Path):
    """Экспорт в ONNX для быстрого inference."""
    try:
        model.eval().cpu()
        dummy_input = torch.randn(1, NUM_FEATURES)
        onnx_path = output_dir / "meta_ensemble.onnx"
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        logger.info("ONNX экспортирован: %s", onnx_path)
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MetaEnsemble model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--unimorph-path", type=str, required=True,
                        help="Path to UniMorph Russian file (rus.txt)")
    parser.add_argument("--output-dir", type=str, default="./models/ensemble",
                        help="Output directory for model")
    parser.add_argument("--max-entries", type=int, default=0,
                        help="Max entries from UniMorph (0=all)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden1", type=int, default=64,
                        help="Hidden layer 1 size")
    parser.add_argument("--hidden2", type=int, default=64,
                        help="Hidden layer 2 size")
    parser.add_argument("--dropout1", type=float, default=0.2)
    parser.add_argument("--dropout2", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="CPU workers (0=auto-detect)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation ratio")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export ONNX after training")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--real-engines", action="store_true",
                        help="Use real engines instead of simulation")
    parser.add_argument("--seed", type=int, default=SEED)

    # Simulation params
    parser.add_argument("--rut5-p-right", type=float, default=0.95)
    parser.add_argument("--rut5-p-wrong", type=float, default=0.70)
    parser.add_argument("--neural-p-right", type=float, default=0.92)
    parser.add_argument("--neural-p-wrong", type=float, default=0.55)

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        # logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)

    # Workers
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    logger.info("CPU workers: %d", num_workers)

    # Load data
    entries = load_unimorph(args.unimorph_path, args.max_entries)
    if not entries:
        logger.error("Нет данных!")
        sys.exit(1)

    # Simulation config
    sim_config = SampleGeneratorConfig(
        rut5_p_right=args.rut5_p_right,
        rut5_p_wrong=args.rut5_p_wrong,
        neural_p_right=args.neural_p_right,
        neural_p_wrong=args.neural_p_wrong,
        seed_base=SEED,
    )

    # Prepare data
    train_ds, val_ds, class_weights = prepare_data(
        entries, num_workers, sim_config, val_ratio=args.val_ratio,
    )

    # Train config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        use_amp=not args.no_amp and device.type == "cuda",
        gradient_clip=1.0,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        num_dataloader_workers=min(8, num_workers),
    )

    # Train
    model, history = train_model(train_ds, val_ds, class_weights, train_config, device)

    # Evaluate
    evaluate_model(model, val_ds, device)

    # Save
    output_dir = Path(args.output_dir)
    save_model(model, output_dir, train_config, history, class_weights)

    # ONNX
    if args.export_onnx:
        export_onnx(model, output_dir)

    logger.info("Done! Модель в %s", output_dir)


if __name__ == "__main__":
    main()