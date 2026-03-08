"""
AnimacyClassifier v2 — runtime aligned with training pipeline.

Загружает модель, обученную train_animacy.py:
  models/animacy/
  ├── animacy_classifier.pt   — AnimacyMLP (state dict saved via torch.save(model))
  └── config.json             — гиперпараметры

Feature pipeline (должен совпадать с train_animacy.py):
  [navec_embedding(300d) ; suffix_hash(30d)] → MLP → sigmoid → anim/inan
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Optional
from pathlib import Path
from ..core.enums import Animacy

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 300
SUFFIX_DIM = 30


def _suffix_features(word: str, dim: int = SUFFIX_DIM) -> np.ndarray:
    """Char-hash суффиксов — ИДЕНТИЧНО train_animacy.suffix_features()."""
    features = np.zeros(dim, dtype=np.float32)
    low = word.lower()
    for length in range(1, 6):
        if len(low) < length:
            break
        suffix = low[-length:]
        h = hash(suffix) % dim
        features[h] += 1.0 / length
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


class AnimacyClassifier:
    """
    Бинарный классификатор одушевлённости.

    Использует Navec-эмбеддинги + суффиксные char-level признаки.
    При отсутствии модели или Navec — суффиксная эвристика.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self._model_path = Path(model_path) if model_path else None
        self._device_pref = device
        self._classifier = None
        self._navec = None
        self._device = "cpu"
        self._available = False
        self._init_attempted = False

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_load()
        return self._available

    def _try_load(self):
        self._init_attempted = True
        if not self._model_path or not self._model_path.exists():
            return

        try:
            import sys
            import torch
            from russian_declension.gpu.training.animacy import AnimacyMLP

            torch.serialization.add_safe_globals([AnimacyMLP])
            sys.modules['__main__'].AnimacyMLP = AnimacyMLP

            if self._device_pref == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._device_pref

            pt_path = self._model_path / "animacy_classifier.pt"
            if not pt_path.exists():
                logger.warning("AnimacyClassifier: файл %s не найден.", pt_path)
                return

            self._classifier = torch.load(str(pt_path), map_location=self._device,
                                           weights_only=False)
            self._classifier.eval()

            # Загрузка Navec (опционально, улучшает точность)
            self._load_navec()

            self._available = True
            logger.info("AnimacyClassifier загружен на %s (navec: %s).",
                        self._device, "да" if self._navec else "нет")
        except Exception as exc:
            logger.warning("AnimacyClassifier: %s", exc)

    def _load_navec(self):
        """Попытка загрузить Navec embeddings."""
        try:
            from navec import Navec
            # Ищем файл рядом с моделью или в текущей директории
            for search_path in [self._model_path, Path(".")]:
                for name in ["navec_hudlit_v1_12B_500K_300d_100q.tar"]:
                    p = search_path / name
                    if p.exists():
                        self._navec = Navec.load(str(p))
                        return
            # Пробуем глобальную загрузку
            self._navec = Navec.load("navec_hudlit_v1_12B_500K_300d_100q.tar")
        except Exception:
            self._navec = None

    def _get_embedding(self, word: str) -> np.ndarray:
        """Navec-эмбеддинг — ИДЕНТИЧНО train_animacy.get_word_embedding()."""
        if self._navec is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        try:
            low = word.lower()
            for w in (word, low):
                if w in self._navec.vocab:
                    idx = self._navec.vocab[w]
                    return self._navec.pq.unpack(idx).astype(np.float32)
        except Exception:
            pass
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def predict(self, word: str) -> tuple[Animacy, float]:
        """Предсказать одушевлённость. Returns: (Animacy, confidence)."""
        if self._available and self._classifier:
            return self._predict_neural(word)
        return self._predict_heuristic(word)

    def _predict_neural(self, word: str) -> tuple[Animacy, float]:
        import torch

        emb = self._get_embedding(word)
        suf = _suffix_features(word)
        features = np.concatenate([emb, suf])
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logit = self._classifier(tensor)
            prob = torch.sigmoid(logit).item()

        if prob > 0.5:
            return (Animacy.ANIMATE, prob)
        return (Animacy.INANIMATE, 1.0 - prob)

    def _predict_heuristic(self, word: str) -> tuple[Animacy, float]:
        """Суффиксная эвристика (fallback)."""
        low = word.lower()
        for suf in ("тель","ник","чик","щик","ист","лог","граф","навт",
                     "вед","ёр","ер","ор","арь","ец"):
            if low.endswith(suf):
                return (Animacy.ANIMATE, 0.7)
        for suf in ("ость","ение","ание","ство","тие","ция","зия","сия",
                     "ура","мент","тор","ика"):
            if low.endswith(suf):
                return (Animacy.INANIMATE, 0.7)
        return (Animacy.INANIMATE, 0.5)
