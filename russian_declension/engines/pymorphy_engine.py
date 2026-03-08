"""
PymorphyEngine — основной engine склонения на базе pymorphy3.

Расширен для поддержки всех частей речи (NOUN, ADJF, PRTF, NUMR, NPRO),
что необходимо для фразового движка (Фаза 3): при склонении фразы
«красивый большой дом» нужно склонять и прилагательные, и причастия.
"""

from __future__ import annotations

import logging
from typing import Optional

from pymorphy3 import MorphAnalyzer
from pymorphy3.analyzer import Parse

from ..core.enums import Case, Gender, Number, Animacy, SpecialGroup
from ..core.models import MorphInfo, InflectionResult, FullParadigm
from ..core.interfaces import IDeclensionEngine

logger = logging.getLogger(__name__)

# ── Маппинг граммем OpenCorpora → наши Enum'ы ─────────────────────

_GENDER_MAP: dict[str, Gender] = {
    "masc": Gender.MASCULINE, "femn": Gender.FEMININE,
    "neut": Gender.NEUTER, "Ms-f": Gender.COMMON,
}

_ANIMACY_MAP: dict[str, Animacy] = {
    "anim": Animacy.ANIMATE, "inan": Animacy.INANIMATE,
}

_CASE_MAP: dict[str, Case] = {
    "nomn": Case.NOMINATIVE, "gent": Case.GENITIVE,
    "datv": Case.DATIVE, "accs": Case.ACCUSATIVE,
    "ablt": Case.INSTRUMENTAL, "loct": Case.PREPOSITIONAL,
    "loc2": Case.LOCATIVE, "gen2": Case.PARTITIVE,
}

_NUMBER_MAP: dict[str, Number] = {
    "sing": Number.SINGULAR, "plur": Number.PLURAL,
}

_SPECIAL_MAP: dict[str, SpecialGroup] = {
    "Fixd": SpecialGroup.INDECLINABLE, "Pltm": SpecialGroup.PLURALIA_TANTUM,
    "Sgtm": SpecialGroup.SINGULARIA_TANTUM, "Subx": SpecialGroup.SUBSTANTIVIZED,
    "Name": SpecialGroup.PROPER_NAME, "Surn": SpecialGroup.SURNAME,
    "Patr": SpecialGroup.PATRONYMIC, "Geox": SpecialGroup.TOPONYM,
}

_ALL_CASES = [
    Case.NOMINATIVE, Case.GENITIVE, Case.DATIVE,
    Case.ACCUSATIVE, Case.INSTRUMENTAL, Case.PREPOSITIONAL,
]
_ALL_NUMBERS = [Number.SINGULAR, Number.PLURAL]

# Части речи, которые склоняются
_DECLINABLE_POS = {"NOUN", "ADJF", "PRTF", "NUMR", "NPRO", "ADJS"}


class PymorphyEngine(IDeclensionEngine):
    """
    Engine склонения на базе pymorphy3 + словаря OpenCorpora.
    Покрывает ~95% русской лексики. Поддерживает существительные,
    прилагательные, причастия, числительные, местоимения.
    """

    def __init__(self, lang: str = "ru"):
        self._lang = lang
        self._morph: Optional[MorphAnalyzer] = None

    @property
    def name(self) -> str:
        return "pymorphy"

    @property
    def confidence_threshold(self) -> float:
        return 0.4

    @property
    def morph(self) -> MorphAnalyzer:
        if self._morph is None:
            self._morph = MorphAnalyzer(lang=self._lang)
        return self._morph

    # ── Основной метод: склонение ────────────────────────────────

    def inflect(
        self, word: str, target_case: Case,
        target_number: Optional[Number] = None,
        context: Optional[str] = None,
    ) -> Optional[InflectionResult]:
        parse = self._best_parse(word)
        if parse is None:
            return None

        morph_info = self._parse_to_morph_info(parse)

        if morph_info.is_indeclinable:
            return InflectionResult(
                word=word, inflected_form=parse.word,
                target_case=target_case,
                target_number=target_number or Number.SINGULAR,
                morph_info=morph_info, engine=self.name,
                confidence=0.99, warnings=["indeclinable"],
            )

        target_grammemes = {target_case.value}

        if target_number is not None:
            if morph_info.is_pluralia_tantum and target_number == Number.SINGULAR:
                return InflectionResult(
                    word=word, inflected_form=parse.word,
                    target_case=target_case, target_number=target_number,
                    morph_info=morph_info, engine=self.name,
                    confidence=0.95, warnings=["pluralia_tantum_no_singular"],
                )
            target_grammemes.add(target_number.value)
        else:
            current_number = self._extract_number(parse)
            if current_number:
                target_grammemes.add(current_number.value)

        inflected = parse.inflect(frozenset(target_grammemes))

        if inflected is None:
            return InflectionResult(
                word=word, inflected_form=word,
                target_case=target_case,
                target_number=target_number or Number.SINGULAR,
                morph_info=morph_info, engine=self.name,
                confidence=0.1, warnings=["inflection_failed"],
            )

        confidence = self._compute_confidence(parse)
        inflected_word = self._preserve_case(word, inflected.word)

        return InflectionResult(
            word=word, inflected_form=inflected_word,
            target_case=target_case,
            target_number=target_number or self._extract_number(parse) or Number.SINGULAR,
            morph_info=morph_info, engine=self.name,
            confidence=confidence,
        )

    def inflect_with_agreement(
        self, word: str, target_case: Case,
        gender: Optional[Gender] = None,
        number: Optional[Number] = None,
        animacy: Optional[Animacy] = None,
    ) -> Optional[str]:
        """
        Склонить слово с принудительным согласованием по роду/числу/одушевлённости.
        Используется фразовым движком для прилагательных и причастий.
        """
        parse = self._best_parse(word)
        if parse is None:
            return None

        target = {target_case.value}
        if number:
            target.add(number.value)
        if gender and number != Number.PLURAL:
            # Род значим только в единственном числе
            target.add(gender.value)
        if animacy:
            target.add(animacy.value)

        inflected = parse.inflect(frozenset(target))
        if inflected is None:
            # Попробуем без одушевлённости
            target.discard("anim")
            target.discard("inan")
            inflected = parse.inflect(frozenset(target))

        if inflected is None:
            return None
        return self._preserve_case(word, inflected.word)

    # ── Морфологический анализ ───────────────────────────────────

    def analyze(self, word: str) -> Optional[MorphInfo]:
        parse = self._best_parse(word)
        if parse is None:
            return None
        return self._parse_to_morph_info(parse)

    def analyze_all(self, word: str) -> list:
        """Вернуть все гипотезы парсинга от pymorphy3 (для фразового движка)."""
        return self.morph.parse(word)

    # ── Полная парадигма ─────────────────────────────────────────

    def paradigm(self, word: str) -> Optional[FullParadigm]:
        parse = self._best_parse(word, noun_only=True)
        if parse is None:
            return None

        morph_info = self._parse_to_morph_info(parse)
        forms: dict[str, Optional[str]] = {}
        numbers = _ALL_NUMBERS
        if morph_info.is_pluralia_tantum:
            numbers = [Number.PLURAL]

        for case in _ALL_CASES:
            for number in numbers:
                key = FullParadigm.make_key(case, number)
                if morph_info.is_indeclinable:
                    forms[key] = parse.word
                    continue
                inflected = parse.inflect(frozenset({case.value, number.value}))
                if inflected is not None:
                    forms[key] = self._preserve_case(word, inflected.word)
                else:
                    forms[key] = None

        return FullParadigm(
            word=word, morph_info=morph_info, forms=forms, engine=self.name,
        )

    # ── Healthcheck ──────────────────────────────────────────────

    def healthcheck(self) -> bool:
        try:
            return len(self.morph.parse("тест")) > 0
        except Exception:
            return False

    # ── Внутренние методы ────────────────────────────────────────

    def _best_parse(self, word: str, noun_only: bool = False) -> Optional[Parse]:
        """
        Найти лучший парсинг для слова.
        Если noun_only=True, ищем только NOUN (для парадигм).
        Иначе ищем любую склоняемую часть речи.
        """
        parses = self.morph.parse(word)
        if not parses:
            return None

        if noun_only:
            candidates = [p for p in parses if "NOUN" in p.tag]
        else:
            candidates = [p for p in parses if any(pos in p.tag for pos in _DECLINABLE_POS)]

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.score, reverse=True)
        return candidates[0]

    def _parse_to_morph_info(self, parse: Parse) -> MorphInfo:
        tag = parse.tag
        tag_set = set(str(tag).replace(" ", ",").split(","))

        gender = next((g for gr, g in _GENDER_MAP.items() if gr in tag_set), None)
        animacy = next((a for gr, a in _ANIMACY_MAP.items() if gr in tag_set), None)
        number = self._extract_number(parse)
        case = self._extract_case(parse)
        special_groups = [sg for gr, sg in _SPECIAL_MAP.items() if gr in tag_set]
        is_noun = "NOUN" in tag_set

        return MorphInfo(
            lemma=parse.normal_form, gender=gender, animacy=animacy,
            number=number, case=case, special_groups=special_groups,
            is_noun=is_noun, score=parse.score,
        )

    @staticmethod
    def _extract_number(parse: Parse) -> Optional[Number]:
        tag_str = str(parse.tag)
        for gr, n in _NUMBER_MAP.items():
            if gr in tag_str:
                return n
        return None

    @staticmethod
    def _extract_case(parse: Parse) -> Optional[Case]:
        tag_str = str(parse.tag)
        for gr, c in _CASE_MAP.items():
            if gr in tag_str:
                return c
        return None

    def _compute_confidence(self, parse: Parse) -> float:
        methods = parse.methods_stack
        is_dictionary = False
        if methods:
            analyzer_class = methods[0][0].__class__.__name__
            is_dictionary = "Dictionary" in analyzer_class

        if is_dictionary:
            return max(parse.score, 0.85)
        else:
            return min(parse.score * 0.6, 0.6)

    @staticmethod
    def _preserve_case(original: str, inflected: str) -> str:
        """Сохранить регистр первой буквы оригинала."""
        if not original or not inflected:
            return inflected
        if original[0].isupper():
            return inflected[0].upper() + inflected[1:]
        return inflected
