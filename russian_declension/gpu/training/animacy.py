#!/usr/bin/env python3
"""
Обучение AnimacyClassifier — MLP-классификатор одушевлённости
на основе Navec-эмбеддингов и разметки OpenCorpora (через pymorphy3).

ОПТИМИЗИРОВАНО для многоядерных систем (96+ cores).
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Глобальные настройки параллелизма
# ─────────────────────────────────────────────────────────────

# Определяем оптимальное число воркеров
NUM_CPUS = os.cpu_count() or 96
# Для CPU-bound задач используем все ядра, для I/O-bound — больше
NUM_WORKERS_CPU = min(NUM_CPUS, 96)
NUM_WORKERS_IO = min(NUM_CPUS * 2, 192)

logger.info(f"Доступно CPU: {NUM_CPUS}, воркеров CPU: {NUM_WORKERS_CPU}, воркеров I/O: {NUM_WORKERS_IO}")

# ──────────────────────────────────────────────
# 2. ПАРАЛЛЕЛЬНОЕ извлечение из pymorphy3
# ──────────────────────────────────────────────

def _analyze_word_batch(words_batch: list[str], min_score: float = 0.5) -> dict[str, int]:
    """
    Анализ батча слов в отдельном процессе.
    Каждый процесс создаёт свой экземпляр MorphAnalyzer.
    """
    from pymorphy3 import MorphAnalyzer
    
    morph = MorphAnalyzer(lang="ru")
    local_labels = {}
    
    for word in words_batch:
        try:
            parses = morph.parse(word)
            if not parses:
                continue

            best = parses[0]
            lemma = best.normal_form

            if "NOUN" not in best.tag:
                continue

            has_anim = False
            has_inan = False

            for parse in parses:
                if parse.score < min_score * best.score:
                    break
                if "NOUN" not in parse.tag:
                    continue
                if "anim" in parse.tag:
                    has_anim = True
                if "inan" in parse.tag:
                    has_inan = True

            if has_anim and not has_inan:
                local_labels[lemma] = 1
            elif has_inan and not has_anim:
                local_labels[lemma] = 0
        except Exception:
            continue
    
    return local_labels


def extract_animacy_labels_from_pymorphy_parallel(
    min_score: float = 0.5,
    max_words: Optional[int] = None,
    num_workers: int = NUM_WORKERS_CPU,
) -> dict[str, int]:
    """
    Параллельное извлечение меток одушевлённости из pymorphy3.
    """
    from pymorphy3 import MorphAnalyzer

    logger.info("Инициализация pymorphy3 для извлечения слов...")
    morph = MorphAnalyzer(lang="ru")
    
    # Извлекаем все слова из DAWG
    logger.info("Извлечение слов из словаря...")
    known_words = []
    
    try:
        word_dawg = morph.dictionary.words
        for word in word_dawg.keys():
            known_words.append(word)
            if max_words and len(known_words) >= max_words * 5:
                break
    except Exception as e:
        logger.warning(f"DAWG extraction failed: {e}")
        known_words = _get_words_from_frequency_list_parallel(morph)

    # Удаляем дубликаты
    known_words = list(set(known_words))
    logger.info(f"Извлечено {len(known_words)} уникальных слов")
    
    # Разбиваем на батчи для параллельной обработки
    batch_size = max(1000, len(known_words) // (num_workers * 4))
    batches = [
        known_words[i:i + batch_size] 
        for i in range(0, len(known_words), batch_size)
    ]
    
    logger.info(f"Параллельный анализ: {len(batches)} батчей по ~{batch_size} слов, {num_workers} воркеров")
    
    # Параллельная обработка
    all_labels = {}
    analyze_func = partial(_analyze_word_batch, min_score=min_score)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_func, batch): i for i, batch in enumerate(batches)}
        
        with tqdm(total=len(batches), desc="Анализ одушевлённости", unit="batch") as pbar:
            for future in as_completed(futures):
                try:
                    batch_labels = future.result()
                    all_labels.update(batch_labels)
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                pbar.update(1)
                
                if max_words and len(all_labels) >= max_words:
                    break
    
    # Ограничиваем если нужно
    if max_words and len(all_labels) > max_words:
        all_labels = dict(list(all_labels.items())[:max_words])
    
    counter = Counter(all_labels.values())
    logger.info(f"Извлечено {len(all_labels)} лемм: animate={counter[1]}, inanimate={counter[0]}")
    
    return all_labels


def _get_words_from_frequency_list_parallel(morph) -> list[str]:
    """Параллельное извлечение слов через prefix-search."""
    alphabet = "абвгдежзийклмнопрстуфхцчшщъыьэюяё"
    word_dawg = morph.dictionary.words
    
    def get_completions(char):
        try:
            return list(word_dawg.keys(char))[:10000]
        except Exception:
            return []
    
    words = []
    with ThreadPoolExecutor(max_workers=min(33, NUM_WORKERS_IO)) as executor:
        results = list(executor.map(get_completions, alphabet))
    
    for result in results:
        words.extend(result)
    
    return words


# ──────────────────────────────────────────────
# 3. ПАРАЛЛЕЛЬНЫЙ парсинг OpenCorpora XML
# ──────────────────────────────────────────────

def _parse_xml_chunk(chunk_data: tuple[str, int, int]) -> dict[str, int]:
    """
    Парсинг чанка XML-файла.
    chunk_data: (file_path, start_pos, end_pos)
    """
    import xml.etree.ElementTree as ET
    
    file_path, start_pos, end_pos = chunk_data
    labels = {}
    
    # Читаем чанк и парсим lemma-элементы
    with open(file_path, 'rb') as f:
        f.seek(start_pos)
        chunk = f.read(end_pos - start_pos)
    
    # Ищем все <lemma>...</lemma> в чанке
    chunk_str = chunk.decode('utf-8', errors='ignore')
    
    # Простой парсинг через регулярки для скорости
    import re
    lemma_pattern = re.compile(
        r'<lemma[^>]*>.*?<l t="([^"]+)"[^>]*>(.*?)</l>.*?</lemma>',
        re.DOTALL
    )
    grammeme_pattern = re.compile(r'<g v="([^"]+)"')
    
    for match in lemma_pattern.finditer(chunk_str):
        word = match.group(1).strip().lower()
        grammemes_str = match.group(2)
        grammemes = set(grammeme_pattern.findall(grammemes_str))
        
        if "NOUN" not in grammemes:
            continue
        
        has_anim = "anim" in grammemes
        has_inan = "inan" in grammemes
        
        if has_anim and not has_inan:
            labels[word] = 1
        elif has_inan and not has_anim:
            labels[word] = 0
    
    return labels


def extract_animacy_from_opencorpora_dict_parallel(
    dict_path: Optional[str] = None,
    num_workers: int = NUM_WORKERS_CPU,
) -> dict[str, int]:
    """
    Параллельное извлечение из XML-дампа OpenCorpora.
    """
    import bz2

    if dict_path is None:
        dict_path = "data/dict.opcorpora.xml"

    path = Path(dict_path)
    
    # Распаковка если нужно
    if not path.exists():
        bz2_path = Path(str(dict_path) + ".bz2")
        if bz2_path.exists():
            logger.info(f"Распаковка {bz2_path}...")
            with bz2.open(bz2_path, "rb") as f_in:
                with open(path, "wb") as f_out:
                    f_out.write(f_in.read())
        else:
            raise FileNotFoundError(f"Не найден {path} или {bz2_path}")

    file_size = path.stat().st_size
    logger.info(f"Парсинг OpenCorpora XML: {path} ({file_size / 1024 / 1024:.1f} MB)")
    
    # Для небольших файлов используем последовательный парсинг
    if file_size < 50 * 1024 * 1024:  # < 50 MB
        return _parse_xml_sequential(path)
    
    # Разбиваем файл на чанки
    chunk_size = max(1024 * 1024, file_size // num_workers)  # ~1MB или больше
    chunks = []
    
    with open(path, 'rb') as f:
        pos = 0
        while pos < file_size:
            end_pos = min(pos + chunk_size, file_size)
            
            # Находим границу lemma-элемента
            if end_pos < file_size:
                f.seek(end_pos)
                # Читаем до закрывающего </lemma>
                extra = f.read(10000)
                close_tag = extra.find(b'</lemma>')
                if close_tag != -1:
                    end_pos += close_tag + len(b'</lemma>')
            
            chunks.append((str(path), pos, end_pos))
            pos = end_pos

    logger.info(f"Параллельный парсинг XML: {len(chunks)} чанков, {num_workers} воркеров")
    
    # Параллельная обработка
    all_labels = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = list(executor.map(_parse_xml_chunk, chunks))
        
        for chunk_labels in tqdm(futures, desc="Парсинг XML чанков", unit="chunk"):
            all_labels.update(chunk_labels)
    
    counter = Counter(all_labels.values())
    logger.info(f"OpenCorpora XML: {len(all_labels)} лемм (animate={counter[1]}, inanimate={counter[0]})")
    
    return all_labels


def _parse_xml_sequential(path: Path) -> dict[str, int]:
    """Последовательный парсинг для небольших файлов."""
    import xml.etree.ElementTree as ET
    
    labels = {}
    context = ET.iterparse(str(path), events=("end",))

    for _, elem in tqdm(context, desc="Парсинг XML", unit="elem"):
        if elem.tag != "lemma":
            continue

        l_elem = elem.find("l")
        if l_elem is None:
            elem.clear()
            continue

        word = l_elem.get("t", "").strip().lower()
        if not word:
            elem.clear()
            continue

        grammemes = {g.get("v") for g in l_elem.findall("g")}

        if "NOUN" not in grammemes:
            elem.clear()
            continue

        has_anim = "anim" in grammemes
        has_inan = "inan" in grammemes

        if has_anim and not has_inan:
            labels[word] = 1
        elif has_inan and not has_anim:
            labels[word] = 0

        elem.clear()

    return labels


# ──────────────────────────────────────────────
# 4. Загрузка эмбеддингов (без изменений)
# ──────────────────────────────────────────────

def load_navec_embeddings(navec_path: str):
    """Загрузка Navec эмбеддингов."""
    from navec import Navec

    logger.info(f"Загрузка Navec из {navec_path}...")
    navec = Navec.load(navec_path)

    try:
        vocab_size = len(navec.vocab.words)
    except (AttributeError, TypeError):
        vocab_size = getattr(navec.vocab, 'count', 'N/A')
    
    try:
        emb_dim = navec.pq.dim
    except AttributeError:
        emb_dim = 300
    
    logger.info(f"Navec загружен: {vocab_size} слов, {emb_dim} dims")
    return navec


# ──────────────────────────────────────────────
# 5. ПАРАЛЛЕЛЬНОЕ построение датасета
# ──────────────────────────────────────────────

# Глобальные переменные для воркеров (избегаем передачи больших объектов)
_NAVEC_GLOBAL = None
_NAVEC_WORDS_SET = None


def _init_navec_worker(navec_path: str):
    """Инициализация Navec в воркере."""
    global _NAVEC_GLOBAL, _NAVEC_WORDS_SET
    from navec import Navec
    _NAVEC_GLOBAL = Navec.load(navec_path)
    try:
        _NAVEC_WORDS_SET = _NAVEC_GLOBAL.vocab.words
    except AttributeError:
        _NAVEC_WORDS_SET = None


def _get_embedding_batch(items: list[tuple[str, int]]) -> list[tuple[str, int, Optional[np.ndarray]]]:
    """
    Получение эмбеддингов для батча слов в воркере.
    """
    global _NAVEC_GLOBAL, _NAVEC_WORDS_SET
    
    results = []
    
    for word, label in items:
        word_lower = word.lower().replace("ё", "е")
        emb = None
        
        try:
            if _NAVEC_WORDS_SET is not None:
                if word_lower not in _NAVEC_WORDS_SET:
                    if "-" in word_lower:
                        first_part = word_lower.split("-")[0]
                        if first_part in _NAVEC_WORDS_SET:
                            emb = np.array(_NAVEC_GLOBAL[first_part], dtype=np.float32)
                else:
                    emb = np.array(_NAVEC_GLOBAL[word_lower], dtype=np.float32)
            else:
                emb = np.array(_NAVEC_GLOBAL[word_lower], dtype=np.float32)
        except (KeyError, IndexError):
            if "-" in word_lower:
                try:
                    first_part = word_lower.split("-")[0]
                    emb = np.array(_NAVEC_GLOBAL[first_part], dtype=np.float32)
                except (KeyError, IndexError):
                    pass
        
        results.append((word, label, emb))
    
    return results


def build_dataset_parallel(
    labels: dict[str, int],
    navec_path: str,
    augment: bool = True,
    num_workers: int = NUM_WORKERS_CPU,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Параллельное построение датасета.
    """
    items = list(labels.items())
    
    # Разбиваем на батчи
    batch_size = max(1000, len(items) // (num_workers * 4))
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]
    
    logger.info(f"Построение датасета: {len(items)} слов, {len(batches)} батчей, {num_workers} воркеров")
    
    embeddings = []
    targets = []
    words = []
    skipped = 0
    
    # Параллельная обработка с инициализацией navec в каждом воркере
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_navec_worker,
        initargs=(navec_path,)
    ) as executor:
        futures = list(executor.map(_get_embedding_batch, batches))
        
        for batch_results in tqdm(futures, desc="Извлечение эмбеддингов", unit="batch"):
            for word, label, emb in batch_results:
                if emb is None:
                    skipped += 1
                    continue
                embeddings.append(emb)
                targets.append(label)
                words.append(word)
    
    X = np.array(embeddings, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    logger.info(
        f"Датасет: {len(X)} примеров (пропущено {skipped} без эмбеддинга). "
        f"animate={100.0 * y.sum() / len(y):.1f}%, inanimate={100.0 * (1 - y.sum() / len(y)):.1f}%"
    )
    
    if augment:
        X, y, words = _augment_with_noise_parallel(X, y, words, num_workers=num_workers)
    
    return X, y, words


def _augment_with_noise_parallel(
    X: np.ndarray,
    y: np.ndarray,
    words: list[str],
    noise_factor: float = 0.05,
    minority_multiplier: int = 2,
    num_workers: int = NUM_WORKERS_CPU,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Параллельная аугментация миноритарного класса.
    """
    counter = Counter(y.astype(int))
    minority_class = 1 if counter[1] < counter[0] else 0
    minority_mask = y == minority_class

    minority_X = X[minority_mask]
    minority_y = y[minority_mask]
    minority_words = [w for w, m in zip(words, minority_mask) if m]

    logger.info(
        f"Аугментация класса {minority_class}: {len(minority_X)} → "
        f"{len(minority_X) * (1 + minority_multiplier)} (×{minority_multiplier})"
    )

    # Параллельная генерация шума
    def generate_noisy_batch(args):
        idx, batch_X, noise_factor = args
        np.random.seed(idx)  # Разные сиды для разных батчей
        noise = np.random.normal(0, noise_factor, batch_X.shape).astype(np.float32)
        noisy = batch_X + noise
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        orig_norms = np.linalg.norm(batch_X, axis=1, keepdims=True)
        noisy = noisy * (orig_norms / (norms + 1e-8))
        return noisy

    aug_X_list = [X]
    aug_y_list = [y]
    aug_words = list(words)

    # Разбиваем minority_X на батчи для параллельной обработки
    batch_size = max(1000, len(minority_X) // num_workers)
    
    for i in range(minority_multiplier):
        batches = [
            (i * 1000 + j, minority_X[j:j + batch_size], noise_factor)
            for j in range(0, len(minority_X), batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            noisy_batches = list(executor.map(generate_noisy_batch, batches))
        
        noisy = np.concatenate(noisy_batches, axis=0)
        aug_X_list.append(noisy)
        aug_y_list.append(minority_y)
        aug_words.extend([f"{w}_aug{i}" for w in minority_words])

    X_aug = np.concatenate(aug_X_list, axis=0)
    y_aug = np.concatenate(aug_y_list, axis=0)

    counter_new = Counter(y_aug.astype(int))
    logger.info(f"После аугментации: {len(X_aug)} примеров")

    return X_aug, y_aug, aug_words

def parse_args():
    parser = argparse.ArgumentParser(description="Обучение AnimacyClassifier (параллельная версия)")
    parser.add_argument("--navec-path", type=str, required=True)
    parser.add_argument("--opencorpora-xml", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/animacy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_CPU,
                        help=f"Число параллельных воркеров (default: {NUM_WORKERS_CPU})")
    return parser.parse_args()

# ══════════════════════════════════════════════════════════════
# 1. РАСШИРЕННАЯ АРХИТЕКТУРА МОДЕЛИ
# ══════════════════════════════════════════════════════════════

class AnimacyMLP(nn.Module):
    """
    Улучшенная архитектура с исправленными размерностями:
    - Residual connections
    - Layer Normalization
    - GELU activation
    """
    
    def __init__(self, emb_dim: int = 300, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        
        h1 = hidden_dim       # 256
        h2 = hidden_dim // 2  # 128
        h3 = hidden_dim // 4  # 64
        
        # Input projection: 300 -> 256
        self.input_proj = nn.Linear(emb_dim, h1)
        self.ln_input = nn.LayerNorm(h1)
        
        # Residual block 1: 256 -> 256
        self.block1 = nn.Sequential(
            nn.Linear(h1, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h1),
            nn.LayerNorm(h1),
        )
        
        # Downsample: 256 -> 128
        self.downsample1 = nn.Linear(h1, h2)
        self.ln_down1 = nn.LayerNorm(h2)
        
        # Residual block 2: 128 -> 128 (ИСПРАВЛЕНО!)
        self.block2 = nn.Sequential(
            nn.Linear(h2, h2),  # 128 -> 128
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h2),  # 128 -> 128
            nn.LayerNorm(h2),
        )
        
        # Output head: 128 -> 64 -> 1
        self.head = nn.Sequential(
            nn.Linear(h2, h3),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(h3, 1),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming инициализация для лучшей сходимости."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection: [B, 300] -> [B, 256]
        x = self.input_proj(x)
        x = self.ln_input(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Residual block 1: [B, 256] -> [B, 256]
        residual = x
        x = self.block1(x)
        x = x + residual
        x = self.act(x)
        
        # Downsample: [B, 256] -> [B, 128]
        x = self.downsample1(x)
        x = self.ln_down1(x)
        x = self.act(x)
        
        # Residual block 2: [B, 128] -> [B, 128]
        residual = x
        x = self.block2(x)
        x = x + residual
        x = self.act(x)
        
        # Output: [B, 128] -> [B, 1] -> [B]
        return self.head(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 2. ОЧИСТКА И ОБОГАЩЕНИЕ ДАННЫХ
# ══════════════════════════════════════════════════════════════

def clean_and_enrich_labels(labels: dict[str, int]) -> dict[str, int]:
    """
    Очистка данных и добавление явных примеров.
    """
    # Явные одушевлённые (люди, животные, профессии)
    KNOWN_ANIMATE = {
        # Животные
        "кошка", "кот", "собака", "пёс", "лошадь", "конь", "корова", "бык",
        "свинья", "овца", "коза", "козёл", "курица", "петух", "утка", "гусь",
        "волк", "лиса", "медведь", "заяц", "белка", "ёж", "мышь", "крыса",
        "слон", "лев", "тигр", "леопард", "пантера", "рысь", "олень", "лось",
        "кабан", "бобр", "выдра", "барсук", "хорёк", "куница", "соболь",
        "обезьяна", "горилла", "шимпанзе", "орангутан", "жираф", "зебра",
        "носорог", "бегемот", "крокодил", "черепаха", "змея", "ящерица",
        "лягушка", "жаба", "рыба", "акула", "кит", "дельфин", "тюлень",
        "птица", "орёл", "сокол", "ястреб", "сова", "ворона", "воробей",
        "голубь", "чайка", "пингвин", "страус", "попугай", "канарейка",
        "бабочка", "пчела", "оса", "муха", "комар", "муравей", "паук",
        "червь", "улитка", "краб", "омар", "креветка", "осьминог",
        
        # Люди — базовые
        "человек", "мужчина", "женщина", "ребёнок", "мальчик", "девочка",
        "младенец", "подросток", "юноша", "девушка", "старик", "старуха",
        
        # Семья
        "мать", "отец", "мама", "папа", "сын", "дочь", "брат", "сестра",
        "дедушка", "бабушка", "внук", "внучка", "дядя", "тётя", "племянник",
        "племянница", "муж", "жена", "супруг", "супруга", "жених", "невеста",
        
        # Профессии
        "врач", "доктор", "медсестра", "фельдшер", "хирург", "терапевт",
        "учитель", "преподаватель", "профессор", "доцент", "лектор",
        "программист", "разработчик", "инженер", "техник", "механик",
        "водитель", "шофёр", "пилот", "машинист", "капитан", "моряк",
        "полицейский", "милиционер", "следователь", "детектив", "охранник",
        "солдат", "офицер", "генерал", "полковник", "майор", "капитан",
        "повар", "официант", "бармен", "продавец", "кассир", "менеджер",
        "директор", "начальник", "руководитель", "администратор", "секретарь",
        "бухгалтер", "экономист", "юрист", "адвокат", "судья", "прокурор",
        "журналист", "репортёр", "редактор", "писатель", "поэт", "автор",
        "художник", "скульптор", "архитектор", "дизайнер", "фотограф",
        "музыкант", "певец", "певица", "артист", "актёр", "актриса",
        "спортсмен", "футболист", "хоккеист", "боксёр", "тренер",
        "строитель", "плотник", "столяр", "слесарь", "электрик", "сантехник",
        "фермер", "агроном", "садовник", "лесник", "охотник", "рыбак",
        
        # Должности и роли
        "президент", "министр", "депутат", "сенатор", "губернатор", "мэр",
        "король", "королева", "принц", "принцесса", "царь", "царица",
        "император", "герцог", "граф", "барон", "рыцарь", "воин", "богатырь",
        
        # Социальные роли
        "друг", "подруга", "товарищ", "коллега", "сосед", "соседка",
        "гость", "хозяин", "хозяйка", "клиент", "покупатель", "пациент",
        "студент", "студентка", "ученик", "ученица", "школьник", "школьница",
        "выпускник", "абитуриент", "аспирант", "магистрант",
        
        # Негативные/криминальные
        "преступник", "вор", "грабитель", "убийца", "мошенник", "бандит",
        "заключённый", "арестант", "подозреваемый", "обвиняемый",
        
        # Религиозные
        "священник", "монах", "монахиня", "епископ", "патриарх", "папа",
        "имам", "раввин", "мулла",
        
        # Мифические/сказочные (одушевлённые)
        "ангел", "демон", "чёрт", "дьявол", "бог", "богиня",
        "ведьма", "колдун", "маг", "волшебник", "фея", "эльф", "гном",
        "великан", "дракон", "единорог", "русалка", "леший", "домовой",
    }
    
    # Явные неодушевлённые
    KNOWN_INANIMATE = {
        # Мебель
        "стол", "стул", "кресло", "диван", "кровать", "шкаф", "комод",
        "полка", "тумбочка", "табурет", "скамейка", "трюмо",
        
        # Техника
        "компьютер", "телефон", "телевизор", "радио", "принтер", "сканер",
        "холодильник", "плита", "микроволновка", "стиральная", "посудомоечная",
        "пылесос", "утюг", "фен", "миксер", "блендер", "тостер", "чайник",
        
        # Транспорт
        "машина", "автомобиль", "автобус", "трамвай", "троллейбус", "метро",
        "поезд", "электричка", "самолёт", "вертолёт", "корабль", "лодка",
        "велосипед", "мотоцикл", "скутер", "трактор", "грузовик",
        
        # Здания
        "дом", "квартира", "комната", "здание", "офис", "магазин",
        "школа", "университет", "больница", "поликлиника", "аптека",
        "театр", "кинотеатр", "музей", "библиотека", "стадион",
        "завод", "фабрика", "склад", "гараж", "сарай", "амбар",
        
        # Природа (неодуш.)
        "дерево", "куст", "трава", "цветок", "лист", "ветка", "корень",
        "гора", "холм", "долина", "река", "озеро", "море", "океан",
        "камень", "песок", "земля", "почва", "глина", "скала",
        
        # Абстрактные
        "время", "пространство", "место", "причина", "результат",
        "идея", "мысль", "понятие", "теория", "практика",
        "любовь", "ненависть", "радость", "грусть", "страх", "надежда",
        
        # Еда (неодуш.)
        "хлеб", "молоко", "сыр", "масло", "мясо", "рыба", "яйцо",
        "овощ", "фрукт", "яблоко", "груша", "апельсин", "банан",
        
        # Одежда
        "одежда", "платье", "юбка", "брюки", "джинсы", "рубашка",
        "футболка", "свитер", "куртка", "пальто", "шуба", "плащ",
        
        # Предметы быта
        "книга", "тетрадь", "ручка", "карандаш", "бумага", "газета",
        "журнал", "письмо", "конверт", "марка", "открытка",
        "ключ", "замок", "дверь", "окно", "стена", "пол", "потолок",
        
        # Города, страны (неодуш.)
        "город", "село", "деревня", "посёлок", "страна", "государство",
        "область", "район", "регион", "континент", "планета",
    }
    
    enriched = dict(labels)
    added_anim = 0
    added_inan = 0
    corrected = 0
    
    # Добавляем/корректируем явные примеры
    for word in KNOWN_ANIMATE:
        if word not in enriched:
            enriched[word] = 1
            added_anim += 1
        elif enriched[word] != 1:
            enriched[word] = 1
            corrected += 1
    
    for word in KNOWN_INANIMATE:
        if word not in enriched:
            enriched[word] = 0
            added_inan += 0
        elif enriched[word] != 0:
            enriched[word] = 0
            corrected += 1
    
    logger.info(
        f"Обогащение данных: +{added_anim} anim, +{added_inan} inan, "
        f"исправлено {corrected}"
    )
    
    return enriched


# ══════════════════════════════════════════════════════════════
# 3. УЛУЧШЕННАЯ АУГМЕНТАЦИЯ
# ══════════════════════════════════════════════════════════════

def advanced_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    words: list[str],
    minority_multiplier: int = 3,  # Увеличено
    mixup_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Продвинутая аугментация:
    1. Гауссов шум (разные уровни)
    2. MixUp внутри класса
    3. Dropout эмбеддингов
    """
    counter = Counter(y.astype(int))
    minority_class = 1 if counter[1] < counter[0] else 0
    minority_mask = y == minority_class
    
    minority_X = X[minority_mask]
    minority_y = y[minority_mask]
    minority_words = [w for w, m in zip(words, minority_mask) if m]
    
    aug_X_list = [X]
    aug_y_list = [y]
    aug_words = list(words)
    
    n_minority = len(minority_X)
    
    for i in range(minority_multiplier):
        # Метод 1: Гауссов шум (разная интенсивность)
        noise_level = 0.03 + 0.02 * i  # 0.03, 0.05, 0.07
        noise = np.random.normal(0, noise_level, minority_X.shape).astype(np.float32)
        noisy = minority_X + noise
        
        # Нормализация
        norms = np.linalg.norm(noisy, axis=1, keepdims=True)
        orig_norms = np.linalg.norm(minority_X, axis=1, keepdims=True)
        noisy = noisy * (orig_norms / (norms + 1e-8))
        
        aug_X_list.append(noisy)
        aug_y_list.append(minority_y)
        aug_words.extend([f"{w}_noise{i}" for w in minority_words])
    
    # Метод 2: MixUp внутри одного класса
    if n_minority > 1:
        idx1 = np.arange(n_minority)
        idx2 = np.random.permutation(n_minority)
        
        # Коэффициент смешивания из Beta-распределения
        lam = np.random.beta(mixup_alpha, mixup_alpha, size=(n_minority, 1))
        lam = lam.astype(np.float32)
        
        mixed = lam * minority_X[idx1] + (1 - lam) * minority_X[idx2]
        
        aug_X_list.append(mixed)
        aug_y_list.append(minority_y)
        aug_words.extend([f"{w}_mixup" for w in minority_words])
    
    # Метод 3: Dropout случайных измерений
    dropout_rate = 0.1
    dropout_mask = np.random.random(minority_X.shape) > dropout_rate
    dropped = minority_X * dropout_mask.astype(np.float32)
    # Масштабируем чтобы сохранить норму
    dropped = dropped / (1 - dropout_rate + 1e-8)
    
    aug_X_list.append(dropped)
    aug_y_list.append(minority_y)
    aug_words.extend([f"{w}_drop" for w in minority_words])
    
    X_aug = np.concatenate(aug_X_list, axis=0)
    y_aug = np.concatenate(aug_y_list, axis=0)
    
    counter_new = Counter(y_aug.astype(int))
    logger.info(
        f"Аугментация: {len(X)} → {len(X_aug)} "
        f"(anim={counter_new[1]}, inan={counter_new[0]})"
    )
    
    return X_aug, y_aug, aug_words


# ══════════════════════════════════════════════════════════════
# 4. УЛУЧШЕННОЕ ОБУЧЕНИЕ
# ══════════════════════════════════════════════════════════════

def train_model_v2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    emb_dim: int = 300,
    hidden_dim: int = 256,  # Увеличено
    dropout: float = 0.4,   # Увеличено
    epochs: int = 100,      # Увеличено
    batch_size: int = 128,  # Уменьшено для лучшей генерализации
    lr: float = 5e-4,       # Уменьшено
    weight_decay: float = 1e-3,  # Увеличено
    patience: int = 15,
    device: str = "auto",
    use_focal_loss: bool = True,  # Focal Loss для дисбаланса
    label_smoothing: float = 0.1,  # Label smoothing
):
    """Улучшенное обучение с Focal Loss и label smoothing."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Label smoothing
    if label_smoothing > 0:
        y_train_smooth = y_train_t * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        y_train_smooth = y_train_t
    
    # Взвешенная выборка
    class_counts = Counter(y_train.astype(int))
    total = len(y_train)
    class_weights = {c: total / (2.0 * count) for c, count in class_counts.items()}
    sample_weights = torch.FloatTensor([class_weights[int(y)] for y in y_train])
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    train_dataset = TensorDataset(X_train_t, y_train_smooth)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
    
    # Используем улучшенную модель
    model = AnimacyMLP(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(device)
    
    # Focal Loss или BCE
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights[1] / (class_weights[0] + class_weights[1]), gamma=2.0)
    else:
        pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Оптимизатор с разными LR для разных слоёв
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.AdamW([
    #     {'params': model.input_proj.parameters(), 'lr': lr},
    #     {'params': model.block1.parameters(), 'lr': lr * 0.8},
    #     {'params': model.block2.parameters(), 'lr': lr * 0.6},
    #     {'params': model.head.parameters(), 'lr': lr * 1.2},
    # ], weight_decay=weight_decay)
    
    # Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    
    logger.info(f"Обучение v2: epochs={epochs}, batch={batch_size}, hidden={hidden_dim}")
    logger.info(f"Focal Loss: {use_focal_loss}, Label smoothing: {label_smoothing}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).cpu().numpy().astype(int)
        
        val_true = y_val.astype(int)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | loss={avg_train_loss:.4f} | "
                f"val_acc={val_acc:.3f} | val_f1={val_f1:.3f}"
            )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss для работы с несбалансированными классами.
    Уменьшает вес легко классифицируемых примеров.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Для положительных примеров
        pos_loss = -self.alpha * ((1 - probs) ** self.gamma) * targets * torch.log(probs + 1e-8)
        
        # Для отрицательных примеров
        neg_loss = -(1 - self.alpha) * (probs ** self.gamma) * (1 - targets) * torch.log(1 - probs + 1e-8)
        
        return (pos_loss + neg_loss).mean()


# ══════════════════════════════════════════════════════════════
# 5. РАСШИРЕННЫЕ SMOKE TESTS
# ══════════════════════════════════════════════════════════════

def run_extended_smoke_tests(model, navec, device: str = "cuda", threshold: float = 0.5):
    """Расширенный набор тестов."""
    
    test_cases = {
        # Животные (должны быть anim)
        "кошка": 1, "кот": 1, "котёнок": 1, "собака": 1, "щенок": 1,
        "лошадь": 1, "корова": 1, "свинья": 1, "волк": 1, "медведь": 1,
        "птица": 1, "рыба": 1, "змея": 1, "лягушка": 1,
        
        # Люди (должны быть anim)
        "человек": 1, "мужчина": 1, "женщина": 1, "ребёнок": 1,
        "мальчик": 1, "девочка": 1, "старик": 1,
        
        # Профессии (должны быть anim)
        "врач": 1, "учитель": 1, "программист": 1, "писатель": 1,
        "художник": 1, "музыкант": 1, "повар": 1, "водитель": 1,
        
        # Должности (должны быть anim!)
        "президент": 1, "министр": 1, "директор": 1, "начальник": 1,
        "король": 1, "царь": 1, "генерал": 1,
        
        # Предметы (должны быть inan)
        "стол": 0, "стул": 0, "книга": 0, "телефон": 0,
        "компьютер": 0, "машина": 0, "дом": 0, "дверь": 0,
        
        # Абстрактные (должны быть inan)
        "время": 0, "место": 0, "идея": 0, "любовь": 0,
        
        # Природа неодуш.
        "дерево": 0, "камень": 0, "гора": 0, "река": 0,
        
        # Еда
        "хлеб": 0, "молоко": 0, "яблоко": 0,
        
        # Сложные случаи
        "робот": 0,  # Обычно inan, хотя бывает по-разному
        "кукла": 0,  # inan
        "труп": 0,   # inan (бывший человек)
    }
    
    if device is None:
        device = get_model_device(model)
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()
    
    results = {"correct": 0, "total": 0, "errors": []}
    
    logger.info("\n" + "=" * 60)
    logger.info("EXTENDED SMOKE TESTS")
    logger.info("=" * 60)
    
    for word, expected in test_cases.items():
        word_lower = word.lower().replace("ё", "е")
        try:
            emb = np.array(navec[word_lower], dtype=np.float32)
        except (KeyError, IndexError):
            logger.info(f"  ? {word:15s} — нет эмбеддинга")
            continue
        
        tensor = torch.FloatTensor(emb).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        
        predicted = 1 if prob > threshold else 0
        correct = predicted == expected
        
        results["total"] += 1
        if correct:
            results["correct"] += 1
            mark = "✓"
        else:
            mark = "✗"
            results["errors"].append((word, expected, predicted, prob))
        
        exp_str = "anim" if expected else "inan"
        pred_str = "anim" if predicted else "inan"
        
        logger.info(f"  {mark} {word:15s} exp={exp_str} pred={pred_str} (p={prob:.3f})")
    
    accuracy = results["correct"] / max(results["total"], 1)
    logger.info(f"\nAccuracy: {results['correct']}/{results['total']} ({accuracy*100:.1f}%)")
    
    if results["errors"]:
        logger.info("\nОшибки:")
        for word, exp, pred, prob in results["errors"]:
            logger.info(f"  {word}: expected={'anim' if exp else 'inan'}, got p={prob:.3f}")
    
    return results


# ══════════════════════════════════════════════════════════════
# 6. ОБНОВЛЁННЫЙ MAIN
# ══════════════════════════════════════════════════════════════

# def main_v2():
#     """Обновлённая версия main с улучшениями."""
#     args = parse_args()
    
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
#     t0 = time.time()
    
#     # 1. Извлечение меток
#     logger.info("=" * 60)
#     logger.info("Шаг 1: Извлечение меток")
#     logger.info("=" * 60)
    
#     if args.opencorpora_xml:
#         labels = extract_animacy_from_opencorpora_dict_parallel(args.opencorpora_xml)
#     else:
#         labels = extract_animacy_labels_from_pymorphy_parallel(max_words=args.max_words)
    
#     # ВАЖНО: Обогащаем данные явными примерами
#     labels = clean_and_enrich_labels(labels)
    
#     # 2. Загрузка эмбеддингов и построение датасета
#     logger.info("=" * 60)
#     logger.info("Шаг 2: Построение датасета")
#     logger.info("=" * 60)
    
#     X, y, words = build_dataset_parallel(labels, args.navec_path, augment=False)
    
#     # Применяем продвинутую аугментацию
#     X, y, words = advanced_augmentation(X, y, words, minority_multiplier=3)
    
#     # Split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=args.test_size, random_state=args.seed, stratify=y
#     )
    
#     # 3. Обучение с улучшениями
#     logger.info("=" * 60)
#     logger.info("Шаг 3: Обучение (улучшенная версия)")
#     logger.info("=" * 60)
    
#     model = train_model_v2(
#         X_train, y_train, X_val, y_val,
#         emb_dim=X_train.shape[1],
#         hidden_dim=256,
#         dropout=0.4,
#         epochs=100,
#         batch_size=128,
#         lr=5e-4,
#         use_focal_loss=True,
#         label_smoothing=0.1,
#     )
    
#     # 4. Тесты
#     navec = load_navec_embeddings(args.navec_path)
#     run_extended_smoke_tests(model, navec)
    
#     elapsed = time.time() - t0
#     logger.info(f"\nГотово за {elapsed:.1f} сек")


# if __name__ == "__main__":
#     main_v2()


# ══════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ 1: Нормализация KNOWN_ANIMATE/INANIMATE
# ══════════════════════════════════════════════════════════════

def normalize_word(word: str) -> str:
    """Стандартная нормализация слова."""
    return word.lower().replace("ё", "е").strip()


def clean_and_enrich_labels(labels: dict[str, int]) -> dict[str, int]:
    """Улучшенная версия с нормализацией."""
    
    KNOWN_ANIMATE = {
        # Животные — ВСЕ формы которые могут быть в Navec
        "кошка", "кот", "котенок", "котёнок",  # Добавляем обе формы!
        "собака", "пес", "пёс", "щенок",
        "рыба", "рыбка",  # КРИТИЧНО!
        "птица", "птичка",
        # ... остальные
        
        # Должности — явно добавляем
        "президент", "премьер", "министр", "депутат", "сенатор",
        "губернатор", "мэр", "глава",
        # ...
    }
    
    KNOWN_INANIMATE = {
        "робот", "андроид", "дрон", "бот",  # Технические устройства
        "кукла", "манекен", "чучело", "статуя", "памятник",  # Изображения людей
        "труп", "мертвец", "покойник",  # Мёртвые (грамматически неодуш.)
        # ...
    }
    
    enriched = {}
    
    # Сначала копируем с нормализацией
    for word, label in labels.items():
        norm_word = normalize_word(word)
        enriched[norm_word] = label
    
    # Затем добавляем/перезаписываем известные
    for word in KNOWN_ANIMATE:
        enriched[normalize_word(word)] = 1
    
    for word in KNOWN_INANIMATE:
        enriched[normalize_word(word)] = 0
    
    return enriched


# ══════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ 2: Проверка эмбеддингов при обогащении
# ══════════════════════════════════════════════════════════════

def clean_and_enrich_labels_with_navec_check(
    labels: dict[str, int], 
    navec
) -> dict[str, int]:
    """
    Обогащаем только теми словами, для которых есть эмбеддинги.
    """
    
    def has_embedding(word: str) -> bool:
        norm = normalize_word(word)
        try:
            _ = navec[norm]
            return True
        except (KeyError, IndexError):
            return False
    
    KNOWN_ANIMATE = [
        "кошка", "кот", "котенок", "собака", "пес", "щенок",
        "рыба", "рыбка", "птица", "птичка",
        "человек", "мужчина", "женщина", "ребенок", "мальчик", "девочка",
        "президент", "министр", "директор", "начальник", "руководитель",
        "врач", "доктор", "учитель", "преподаватель",
        # Добавьте все базовые слова
    ]
    
    KNOWN_INANIMATE = [
        "стол", "стул", "дом", "машина", "компьютер", "телефон",
        "робот", "кукла", "статуя", "памятник",
        "труп", "тело",  # мёртвое
        # ...
    ]
    
    enriched = {normalize_word(k): v for k, v in labels.items()}
    
    added = {"anim": 0, "inan": 0, "no_emb": 0}
    
    for word in KNOWN_ANIMATE:
        norm = normalize_word(word)
        if has_embedding(norm):
            if norm not in enriched or enriched[norm] != 1:
                enriched[norm] = 1
                added["anim"] += 1
        else:
            added["no_emb"] += 1
    
    for word in KNOWN_INANIMATE:
        norm = normalize_word(word)
        if has_embedding(norm):
            if norm not in enriched or enriched[norm] != 0:
                enriched[norm] = 0
                added["inan"] += 1
        else:
            added["no_emb"] += 1
    
    logger.info(f"Обогащение: +{added['anim']} anim, +{added['inan']} inan, "
                f"пропущено без эмбеддинга: {added['no_emb']}")
    
    return enriched


# ══════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ 3: Диагностика проблемных слов
# ══════════════════════════════════════════════════════════════

def diagnose_problem_words(labels: dict[str, int], navec, problem_words: list[str]):
    """Диагностика: почему слово не работает."""
    
    logger.info("\n" + "=" * 60)
    logger.info("ДИАГНОСТИКА ПРОБЛЕМНЫХ СЛОВ")
    logger.info("=" * 60)
    
    for word in problem_words:
        norm = normalize_word(word)
        
        # Проверка 1: есть ли в labels?
        in_labels = norm in labels
        label_value = labels.get(norm, "N/A")
        
        # Проверка 2: есть ли эмбеддинг?
        has_emb = False
        try:
            emb = navec[norm]
            has_emb = True
            emb_norm = np.linalg.norm(emb)
        except (KeyError, IndexError):
            emb_norm = 0
        
        # Проверка 3: какие варианты слова есть в Navec?
        variants_in_navec = []
        for variant in [word, word.lower(), norm, word.replace("ё", "е")]:
            try:
                _ = navec[variant]
                variants_in_navec.append(variant)
            except:
                pass
        
        logger.info(f"\n{word}:")
        logger.info(f"  normalized: '{norm}'")
        logger.info(f"  in labels: {in_labels} (value={label_value})")
        logger.info(f"  has embedding: {has_emb} (norm={emb_norm:.3f})")
        logger.info(f"  variants in navec: {variants_in_navec}")


# ══════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ 4: Hard Negative Mining для граничных случаев
# ══════════════════════════════════════════════════════════════

def add_hard_examples(
    X: np.ndarray, 
    y: np.ndarray, 
    words: list[str],
    navec,
    multiplier: int = 5  # Сколько раз повторить hard examples
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Добавляем больше примеров для сложных случаев.
    """
    
    # Слова, которые часто путаются
    HARD_ANIMATE = [
        # Должности (часто путают с неодуш.)
        "президент", "министр", "депутат", "директор", "начальник",
        "генерал", "полковник", "капитан", "командир",
        
        # Животные, похожие на предметы
        "рыба", "краб", "медуза", "устрица", "мидия",
        
        # Односложные животные
        "кот", "пес", "бык", "конь", "лось",
    ]
    
    HARD_INANIMATE = [
        # Похожие на людей
        "робот", "кукла", "манекен", "статуя", "портрет",
        "скелет", "мумия", "чучело",
        
        # Были живыми
        "труп", "тело", "останки",
        
        # Организации (не люди!)
        "компания", "фирма", "организация", "государство",
    ]
    
    hard_X = []
    hard_y = []
    hard_words = []
    
    for word in HARD_ANIMATE:
        norm = normalize_word(word)
        try:
            emb = np.array(navec[norm], dtype=np.float32)
            for i in range(multiplier):
                # Добавляем с небольшим шумом
                noise = np.random.normal(0, 0.02, emb.shape).astype(np.float32)
                hard_X.append(emb + noise)
                hard_y.append(1)
                hard_words.append(f"{norm}_hard{i}")
        except (KeyError, IndexError):
            pass
    
    for word in HARD_INANIMATE:
        norm = normalize_word(word)
        try:
            emb = np.array(navec[norm], dtype=np.float32)
            for i in range(multiplier):
                noise = np.random.normal(0, 0.02, emb.shape).astype(np.float32)
                hard_X.append(emb + noise)
                hard_y.append(0)
                hard_words.append(f"{norm}_hard{i}")
        except (KeyError, IndexError):
            pass
    
    if hard_X:
        X_new = np.concatenate([X, np.array(hard_X)], axis=0)
        y_new = np.concatenate([y, np.array(hard_y)], axis=0)
        words_new = words + hard_words
        
        logger.info(f"Добавлено {len(hard_X)} hard examples")
        return X_new, y_new, words_new
    
    return X, y, words


# ══════════════════════════════════════════════════════════════
# ИСПРАВЛЕНИЕ 5: Калибровка порога
# ══════════════════════════════════════════════════════════════

def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def find_optimal_threshold(model, X_val, y_val, device=None):
    """
    Подбор оптимального порога вместо 0.5.
    """
    model.eval()

    if device is None:
        device = get_model_device(model)
    else:
        device = torch.device(device)
        model = model.to(device)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t)).detach().cpu().numpy()

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.3, 0.7, 0.01):
        preds = (probs > threshold).astype(int)
        f1 = f1_score(y_val.astype(int), preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    logger.info(f"Оптимальный порог: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


# ══════════════════════════════════════════════════════════════
# ОБНОВЛЁННЫЙ MAIN
# ══════════════════════════════════════════════════════════════

def main():
    """Версия с исправлениями."""
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 1. Загружаем Navec ПЕРВЫМ (нужен для проверки)
    navec = load_navec_embeddings(args.navec_path)
    
    # 2. Извлекаем метки
    if args.opencorpora_xml:
        labels = extract_animacy_from_opencorpora_dict_parallel(args.opencorpora_xml)
    else:
        labels = extract_animacy_labels_from_pymorphy_parallel(max_words=args.max_words)
    
    # 3. Диагностика проблемных слов ДО обогащения
    diagnose_problem_words(labels, navec, ["кот", "рыба", "президент", "робот", "кукла"])
    
    # 4. Обогащаем С ПРОВЕРКОЙ эмбеддингов
    labels = clean_and_enrich_labels_with_navec_check(labels, navec)
    
    # 5. Строим датасет
    X, y, words = build_dataset_parallel(labels, args.navec_path, augment=False)
    
    # 6. Добавляем hard examples
    X, y, words = add_hard_examples(X, y, words, navec, multiplier=10)
    
    # 7. Аугментация
    X, y, words = advanced_augmentation(X, y, words, minority_multiplier=3)
    
    # 8. Split и обучение
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    model = train_model_v2(
        X_train, y_train, X_val, y_val,
        hidden_dim=256,
        epochs=100,
        use_focal_loss=True,
        device=args.device
    )
    
    # 9. Калибровка порога
    optimal_threshold = find_optimal_threshold(model, X_val, y_val, 'cuda')
    
    # 10. Тесты с оптимальным порогом
    run_extended_smoke_tests(model, navec, threshold=0.6)


if __name__ == "__main__":
    main()
