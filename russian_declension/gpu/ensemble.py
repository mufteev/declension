"""
MetaEnsemble — мета-модель для ансамблирования результатов engine'ов.

Вместо жёсткой цепочки fallback (где каждый engine решает «уверен/передаю»),
мета-модель получает результаты и confidence-скоры от всех engine'ов
и выбирает наилучший. Подход предложен Kuzmenko (2016, HSE).

Входной вектор: [pymorphy_conf, pymorphy_is_dict, rut5_conf, rut5_len_diff,
                 char_transformer_conf, word_suffix_features...]
Выход: вероятность для каждого engine, что его результат правильный.

Обучение: holdout из UniMorph + SynTagRus с ground truth.
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path

from ..core.models import InflectionResult

logger = logging.getLogger(__name__)


class MetaEnsemble:
    """
    Мета-модель для выбора лучшего результата из нескольких engine'ов.

    Использование:
        ensemble = MetaEnsemble(model_path="models/ensemble.pt")
        best = ensemble.select_best([
            result_pymorphy,    # InflectionResult от pymorphy
            result_rut5,        # InflectionResult от ruT5
            result_chartrans,   # InflectionResult от char-transformer
        ], word="кошка")
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self._model_path = Path(model_path) if model_path else None
        self._model = None
        self._available = False
        self._init_attempted = False

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_load()
        return self._available

    def _try_load(self):
        self._init_attempted = True
        if self._model_path and self._model_path.exists():
            try:
                import torch
                self._model = torch.load(
                    self._model_path / "meta_ensemble.pt",
                    map_location="cpu", weights_only=False)
                self._model.eval()
                self._available = True
                logger.info("MetaEnsemble загружен.")
            except Exception as exc:
                logger.warning("MetaEnsemble: %s", exc)

    def select_best(self, results: list[InflectionResult],
                    word: str = "") -> InflectionResult:
        """
        Выбрать лучший результат из списка.

        Если мета-модель доступна — используем нейросетевой выбор.
        Иначе — эвристика: берём результат с наивысшей confidence,
        при равенстве — от более «надёжного» engine.
        """
        if not results:
            raise ValueError("results не может быть пустым")

        if len(results) == 1:
            return results[0]

        if self.is_available:
            return self._select_neural(results, word)
        return self._select_heuristic(results)

    def _select_neural(self, results: list[InflectionResult],
                       word: str) -> InflectionResult:
        """Нейросетевой выбор через мета-модель."""
        import torch

        features = self._extract_features(results, word)
        tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Выбираем engine с наибольшей вероятностью
        best_idx = probs.argmax().item()
        best_idx = min(best_idx, len(results) - 1)

        selected = results[best_idx]
        # Обновляем confidence на основе мета-модели
        selected.confidence = max(selected.confidence, probs[best_idx].item())
        return selected

    def _select_heuristic(self, results: list[InflectionResult]) -> InflectionResult:
        """
        Эвристический выбор (fallback без мета-модели).

        Приоритет engine'ов: pymorphy > rut5 > neural > unknown.
        При одинаковом приоритете — по confidence.
        """
        engine_priority = {"pymorphy": 4, "rut5": 3, "neural": 2, "cache": 5}

        def sort_key(r: InflectionResult) -> tuple:
            priority = engine_priority.get(r.engine, 1)
            return (r.confidence > 0.8, priority, r.confidence)

        results.sort(key=sort_key, reverse=True)
        return results[0]

    def _extract_features(self, results: list[InflectionResult],
                          word: str) -> list[float]:
        """Извлечь вектор признаков для мета-модели."""
        features = []

        # Фиксированный порядок: pymorphy, rut5, neural (заполняем нулями если нет)
        engine_order = ["pymorphy", "rut5", "neural"]
        result_by_engine = {r.engine: r for r in results}

        for eng_name in engine_order:
            r = result_by_engine.get(eng_name)
            if r:
                features.extend([
                    r.confidence,
                    1.0 if r.confidence > 0.8 else 0.0,
                    len(r.inflected_form) / max(len(word), 1),
                    1.0 if r.inflected_form != word else 0.0,
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Суффиксные признаки слова (6 бит на последние 3 символа)
        for i in range(1, 4):
            if len(word) >= i:
                features.append(ord(word[-i]) / 1200.0)
            else:
                features.append(0.0)

        # Длина слова (нормализованная)
        features.append(len(word) / 20.0)

        return features
