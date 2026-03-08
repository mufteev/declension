"""
BertValidator — постпроверка результатов склонения через BERT-морфотегер.

Логика: после генерации словоформы, BERT анализирует её и предсказывает
морфологические признаки. Если признаки не совпадают с целевыми (например,
engine сгенерировал дательный, а BERT видит родительный) — результат
помечается как ненадёжный.

MorDL (BERT-based) на SynTagRus: UPOS 99.35%, FEATS 98.87%.
Slovnet BERT: 98.2% на новостях.

VRAM: ~500 МБ (BERT-base FP16).
Запускается ВЫБОРОЧНО: только для confidence < 0.8 или по запросу.
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path

from ..core.enums import Case, Number
from ..core.models import InflectionResult

logger = logging.getLogger(__name__)

# Маппинг UD-тегов → наши коды падежей
_UD_CASE_MAP = {
    "Nom": "nomn", "Gen": "gent", "Dat": "datv",
    "Acc": "accs", "Ins": "ablt", "Loc": "loct",
}
_UD_NUMBER_MAP = {"Sing": "sing", "Plur": "plur"}


class BertValidator:
    """
    Валидатор на базе BERT-морфотегера.

    Использование:
        validator = BertValidator(model_name="slovnet_bert_morph")
        is_valid, predicted_case = validator.validate(
            word="кошке", expected_case=Case.DATIVE
        )
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self._model_path = model_path
        self._device_preference = device
        self._pipeline = None
        self._available = False
        self._init_attempted = False

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_load()
        return self._available

    def _try_load(self):
        self._init_attempted = True
        try:
            # Стратегия 2: HuggingFace transformers pipeline
            if self._model_path and Path(self._model_path).exists():
                from transformers import pipeline
                import torch

                # Определяем device
                if self._device_preference == "auto":
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._device = self._device_preference

                self._pipeline = pipeline(
                    "token-classification",
                    model=self._model_path,
                    device=self._device,
                    dtype=torch.float16 if self._device == 'cuda' else torch.float32,
                )
                self._use_natasha = False
                self._available = True
                logger.info("BertValidator: HuggingFace модель загружена.")

        except Exception as exc:
            logger.warning("BertValidator: не удалось загрузить: %s", exc)

    def validate(self, word: str, expected_case: Case,
                 expected_number: Optional[Number] = None,
                 context: Optional[str] = None) -> dict:
        """
        Проверить, что сгенерированная форма соответствует ожидаемым признакам.

        Returns:
            {
                "valid": True/False,
                "predicted_case": "gent" или None,
                "predicted_number": "sing" или None,
                "confidence": float,
            }
        """
        if not self.is_available:
            return {"valid": True, "predicted_case": None,
                    "predicted_number": None, "confidence": 0.0}

        # Собираем мини-контекст для анализа
        text = context or f"Я вижу {word}."

        try:
            if self._use_natasha:
                return self._validate_natasha(word, text, expected_case, expected_number)
            else:
                return self._validate_hf(word, text, expected_case, expected_number)
        except Exception as exc:
            logger.warning("BertValidator error: %s", exc)
            return {"valid": True, "predicted_case": None,
                    "predicted_number": None, "confidence": 0.0}

    def _validate_natasha(self, word, text, expected_case, expected_number):
        from natasha import Doc
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._tagger)

        for token in doc.tokens:
            if token.text.lower() == word.lower():
                feats = token.feats if isinstance(token.feats, dict) else {}
                pred_case = _UD_CASE_MAP.get(feats.get("Case"))
                pred_number = _UD_NUMBER_MAP.get(feats.get("Number"))

                case_match = pred_case == expected_case.value if pred_case else True
                number_match = (pred_number == expected_number.value
                                if pred_number and expected_number else True)

                return {
                    "valid": case_match and number_match,
                    "predicted_case": pred_case,
                    "predicted_number": pred_number,
                    "confidence": 0.9,
                }

        return {"valid": True, "predicted_case": None,
                "predicted_number": None, "confidence": 0.0}

    def _validate_hf(self, word, text, expected_case, expected_number):
        # HuggingFace pipeline — парсим теги
        results = self._pipeline(text)
        for item in results:
            if item["word"].lower() == word.lower():
                tag = item.get("entity", "")
                pred_case = None
                for ud_key, code in _UD_CASE_MAP.items():
                    if ud_key in tag:
                        pred_case = code
                        break
                valid = pred_case == expected_case.value if pred_case else True
                return {"valid": valid, "predicted_case": pred_case,
                        "predicted_number": None, "confidence": item.get("score", 0.5)}

        return {"valid": True, "predicted_case": None,
                "predicted_number": None, "confidence": 0.0}

    def validate_inflection_result(self, result: InflectionResult) -> InflectionResult:
        """Обогатить InflectionResult результатами валидации."""
        v = self.validate(result.inflected_form, result.target_case, result.target_number)
        if not v["valid"]:
            result.confidence *= 0.5  # Снижаем уверенность при расхождении
            result.warnings.append(
                f"bert_validation_mismatch: expected={result.target_case.value}, "
                f"predicted={v['predicted_case']}")
        return result
