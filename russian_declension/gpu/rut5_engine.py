"""
RuT5Engine — fine-tuned ruT5 для морфологической генерации.

Это ключевой GPU-компонент Плана 2: ruT5-small (~60M параметров),
fine-tuned на задаче «лемма + признаки → словоформа».

Формат входа модели:
  «inflect: кошка case=gent number=plur»

Формат выхода:
  «кошек»

Преимущество над char-level Transformer: T5 уже «знает» морфологические
закономерности русского из предобучения — fine-tuning лишь активирует
эти знания. Ожидаемая точность на OOV: 94–97% (vs 91–96% у char-level).

VRAM: ~500 МБ (FP16 с KV-cache).
Скорость: ~5–10 мс/слово (single), ~1–2 мс/слово (batch 32+).
"""

from __future__ import annotations
import logging
from typing import Optional
from pathlib import Path

from ..core.enums import Case, Number, Gender, Animacy
from ..core.models import InflectionResult, MorphInfo, FullParadigm
from ..core.interfaces import IDeclensionEngine

logger = logging.getLogger(__name__)

# Маппинг enum → строковые коды для промпта T5
_CASE_CODE = {
    Case.NOMINATIVE: "nomn", Case.GENITIVE: "gent",
    Case.DATIVE: "datv", Case.ACCUSATIVE: "accs",
    Case.INSTRUMENTAL: "ablt", Case.PREPOSITIONAL: "loct",
}
_NUMBER_CODE = {Number.SINGULAR: "sing", Number.PLURAL: "plur"}

_ALL_CASES = [Case.NOMINATIVE, Case.GENITIVE, Case.DATIVE,
              Case.ACCUSATIVE, Case.INSTRUMENTAL, Case.PREPOSITIONAL]
_ALL_NUMBERS = [Number.SINGULAR, Number.PLURAL]


class RuT5Engine(IDeclensionEngine):
    """
    GPU-ускоренный engine на базе fine-tuned ruT5-small.

    Загрузка модели:
      engine = RuT5Engine(model_path="models/rut5-declension")
      result = engine.inflect("кошка", Case.GENITIVE, Number.PLURAL)
      # → InflectionResult(inflected_form="кошек", confidence=0.96)

    Если модель не найдена — engine деградирует в неактивное состояние,
    FallbackChain его пропустит.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self._model_path = Path(model_path) if model_path else None
        self._device_preference = device
        self._model = None
        self._tokenizer = None
        self._device = None
        self._available = False
        self._init_attempted = False

    @property
    def name(self) -> str:
        return "rut5"

    @property
    def confidence_threshold(self) -> float:
        return 0.5  # Выше чем у char-level (0.3), ниже чем у pymorphy (0.4)

    @property
    def is_available(self) -> bool:
        if not self._init_attempted:
            self._try_load()
        return self._available

    def _try_load(self):
        """Загрузка модели с graceful degradation."""
        self._init_attempted = True
        if self._model_path is None or not self._model_path.exists():
            if self._model_path:
                logger.warning("RuT5Engine: путь '%s' не найден.", self._model_path)
            return

        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            # Определяем device
            if self._device_preference == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._device_preference

            logger.info("RuT5Engine: загрузка из '%s' на %s...",
                        self._model_path, self._device)

            self._tokenizer = T5Tokenizer.from_pretrained(str(self._model_path))
            self._model = T5ForConditionalGeneration.from_pretrained(
                str(self._model_path),
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            self._available = True
            logger.info("RuT5Engine: модель загружена (%s).", self._device)

        except Exception as exc:
            logger.error("RuT5Engine: ошибка загрузки: %s", exc)
            self._available = False

    def _build_prompt(self, word: str, case: Case, number: Optional[Number]) -> str:
        """Формирование входной строки для T5."""
        prompt = f"inflect: {word.lower()} case={_CASE_CODE[case]}"
        if number:
            prompt += f" number={_NUMBER_CODE[number]}"
        return prompt

    def inflect(self, word: str, target_case: Case,
                target_number: Optional[Number] = None,
                context: Optional[str] = None) -> Optional[InflectionResult]:
        if not self.is_available:
            return None

        import torch

        prompt = self._build_prompt(word, target_case, target_number)
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,           # Beam search для лучшего качества
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated = self._tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        generated = generated.strip().lower()

        # Оценка confidence через beam scores
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            score = torch.exp(outputs.sequences_scores[0]).item()
            confidence = min(score, 0.99)
        else:
            confidence = 0.7

        # Сохраняем регистр
        if word and word[0].isupper() and generated:
            generated = generated[0].upper() + generated[1:]

        return InflectionResult(
            word=word, inflected_form=generated,
            target_case=target_case,
            target_number=target_number or Number.SINGULAR,
            engine=self.name, confidence=confidence,
        )

    def inflect_batch(self, items: list[tuple[str, Case, Optional[Number]]]) -> list[Optional[InflectionResult]]:
        """Батчевый инференс — ключевое преимущество GPU."""
        if not self.is_available:
            return [None] * len(items)

        import torch

        prompts = [self._build_prompt(w, c, n) for w, c, n in items]
        inputs = self._tokenizer(prompts, return_tensors="pt",
                                 padding=True, truncation=True).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=40,
                num_beams=4, early_stopping=True,
            )

        results = []
        for i, (word, case, number) in enumerate(items):
            gen = self._tokenizer.decode(outputs[i], skip_special_tokens=True).strip().lower()
            if word and word[0].isupper() and gen:
                gen = gen[0].upper() + gen[1:]
            results.append(InflectionResult(
                word=word, inflected_form=gen,
                target_case=case, target_number=number or Number.SINGULAR,
                engine=self.name, confidence=0.7,
            ))
        return results

    def analyze(self, word: str) -> Optional[MorphInfo]:
        return None  # T5 не делает анализ, только генерацию

    def paradigm(self, word: str) -> Optional[FullParadigm]:
        """Сгенерировать полную парадигму батчем — эффективно на GPU."""
        if not self.is_available:
            return None

        items = [(word, c, n) for c in _ALL_CASES for n in _ALL_NUMBERS]
        batch_results = self.inflect_batch(items)

        forms = {}
        for (_, case, number), result in zip(items, batch_results):
            if result:
                key = FullParadigm.make_key(case, number)
                forms[key] = result.inflected_form

        return FullParadigm(
            word=word,
            morph_info=MorphInfo(lemma=word, is_noun=True, score=0.7),
            forms=forms, engine=self.name,
        )

    def healthcheck(self) -> bool:
        return self.is_available
