"""
MetaEnsemble v2 — runtime aligned with training pipeline.

Загружает модель, обученную train_ensemble.py:
  models/ensemble/
  ├── meta_ensemble.pt   — EnsembleMLP (torch.save(model))
  └── config.json        — гиперпараметры

Feature pipeline (ИДЕНТИЧНО train_ensemble.extract_features):
  Per-engine (×3): [confidence, is_high_conf, len_ratio, changed]
  Suffix (×3): [ord(char[-1])/1200, ord(char[-2])/1200, ord(char[-3])/1200]
  Word: [len/20]
  Итого: 16 features → MLP → softmax → выбор engine
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path
from ..core.models import InflectionResult

logger = logging.getLogger(__name__)

ENGINE_ORDER = ["pymorphy", "rut5", "neural"]


class MetaEnsemble:
    """
    Мета-модель: выбирает лучший результат из нескольких engine'ов.

    При отсутствии модели — эвристика по confidence + engine priority.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self._model_path = Path(model_path) if model_path else None
        self._model = None
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
        pt_path = self._model_path / "meta_ensemble.pt"
        if not pt_path.exists():
            return
        try:
            import sys
            import torch
            from russian_declension.gpu.training.train_ensemble import EnsembleMLP

            torch.serialization.add_safe_globals([EnsembleMLP])
            sys.modules['__main__'].EnsembleMLP = EnsembleMLP

            self._device = ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = torch.load(str(pt_path), map_location=self._device,
                                     weights_only=False)
            self._model.eval()
            self._available = True
            logger.info("MetaEnsemble загружен.")
        except Exception as exc:
            logger.warning("MetaEnsemble: %s", exc)

    def select_best(self, results: list[InflectionResult],
                    word: str = "") -> InflectionResult:
        """Выбрать лучший результат из списка."""
        if not results:
            raise ValueError("results пуст")
        if len(results) == 1:
            return results[0]
        if self._available:
            return self._select_neural(results, word)
        return self._select_heuristic(results)

    def _select_neural(self, results: list[InflectionResult],
                       word: str) -> InflectionResult:
        import torch

        features = self._extract_features(results, word)
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Маппинг: индекс softmax → реальный engine из results
        result_by_engine = {r.engine: r for r in results}
        best_result = None
        best_prob = -1.0

        for idx, eng_name in enumerate(ENGINE_ORDER):
            if idx < len(probs):
                p = probs[idx].item()
                r = result_by_engine.get(eng_name)
                if r and p > best_prob:
                    best_prob = p
                    best_result = r

        if best_result is None:
            return self._select_heuristic(results)

        best_result.confidence = max(best_result.confidence, best_prob)
        return best_result

    def _select_heuristic(self, results: list[InflectionResult]) -> InflectionResult:
        """Эвристика: engine priority + confidence."""
        priority = {"cache": 5, "pymorphy": 4, "rut5": 3, "neural": 2}

        def key(r):
            return (r.confidence > 0.8, priority.get(r.engine, 1), r.confidence)

        return max(results, key=key)

    @staticmethod
    def _extract_features(results: list[InflectionResult],
                          word: str) -> list[float]:
        """
        Вектор признаков — ИДЕНТИЧНО train_ensemble.extract_features().

        16 features:
          [pymorphy: conf, hi_conf, len_ratio, changed] ×3 engines
          [suffix_char_1, suffix_char_2, suffix_char_3]
          [word_len_norm]
        """
        features = []
        by_engine = {r.engine: r for r in results}

        for eng_name in ENGINE_ORDER:
            r = by_engine.get(eng_name)
            if r:
                features.extend([
                    r.confidence,
                    1.0 if r.confidence > 0.8 else 0.0,
                    len(r.inflected_form) / max(len(word), 1),
                    1.0 if r.inflected_form != word else 0.0,
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        for i in range(1, 4):
            features.append(ord(word[-i]) / 1200.0 if len(word) >= i else 0.0)

        features.append(len(word) / 20.0)
        return features
