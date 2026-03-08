"""
Модуль склонения русских числительных.

Числительные — самая сложная подсистема: 7+ парадигм, нет единого паттерна.
Ни один открытый инструмент не склоняет их во всех 6 падежах.

Алгоритм:
  1. Разложить число на компоненты (тысячи, сотни, десятки, единицы)
  2. Конвертировать каждый компонент в слово (num2words или собственная таблица)
  3. Склонить каждое слово через pymorphy3
  4. Применить правила согласования числительного с существительным
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.enums import Case, Number, Gender

logger = logging.getLogger(__name__)

# ── Таблицы числительных в именительном падеже ─────────────────────

_ONES = {
    0: "", 1: "один", 2: "два", 3: "три", 4: "четыре",
    5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
    10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать",
    14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать",
    17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать",
}

_ONES_FEM = {1: "одна", 2: "две"}  # Женский род для "одна тысяча", "две тысячи"

_TENS = {
    2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят",
    6: "шестьдесят", 7: "семьдесят", 8: "восемьдесят", 9: "девяносто",
}

_HUNDREDS = {
    1: "сто", 2: "двести", 3: "триста", 4: "четыреста",
    5: "пятьсот", 6: "шестьсот", 7: "семьсот", 8: "восемьсот", 9: "девятьсот",
}


class NumeralEngine:
    """
    Движок склонения числительных с согласованием с существительным.

    Использование:
        engine = NumeralEngine()
        # «двадцати одному рублю»
        result = engine.inflect_numeral(21, Case.DATIVE, unit="рубль", unit_gender=Gender.MASCULINE)
    """

    def __init__(self):
        self._morph = None

    @property
    def morph(self):
        if self._morph is None:
            from pymorphy3 import MorphAnalyzer
            self._morph = MorphAnalyzer(lang="ru")
        return self._morph

    # ── Главный метод ─────────────────────────────────────────────

    def inflect_numeral(
        self,
        number: int,
        target_case: Case,
        unit: Optional[str] = None,
        unit_gender: Gender = Gender.MASCULINE,
    ) -> str:
        """
        Склонить числительное (+ опциональное существительное) в целевой падеж.

        Args:
            number: Целое число (0 .. 999_999_999_999)
            target_case: Целевой падеж
            unit: Единица измерения (напр. «рубль», «штука»)
            unit_gender: Род единицы измерения

        Returns:
            Строка вида «двадцати одному рублю»
        """
        if number == 0:
            word = "ноль"
            inflected_word = self._inflect_word(word, target_case)
            if unit:
                inflected_unit = self._inflect_unit_for_number(unit, number, target_case)
                return f"{inflected_word} {inflected_unit}"
            return inflected_word

        # Разложить число на слова
        words = self._number_to_words(number, unit_gender)

        # Склонить каждое слово числительного в целевой падеж
        if target_case == Case.NOMINATIVE:
            inflected_words = words
        else:
            inflected_words = [self._inflect_word(w, target_case) for w in words]

        result = " ".join(inflected_words)

        # Добавить единицу измерения с правильным согласованием
        if unit:
            inflected_unit = self._inflect_unit_for_number(unit, number, target_case)
            result = f"{result} {inflected_unit}"

        return result

    # ── Число → слова (именительный падеж) ────────────────────────

    def _number_to_words(self, n: int, gender: Gender = Gender.MASCULINE) -> list[str]:
        """Разложить число на список слов в именительном падеже."""
        if n == 0:
            return ["ноль"]
        if n < 0:
            return ["минус"] + self._number_to_words(-n, gender)

        parts: list[str] = []

        # Миллиарды
        if n >= 1_000_000_000:
            billions = n // 1_000_000_000
            parts.extend(self._group_to_words(billions, Gender.MASCULINE))
            b_form = self._noun_form_by_number(billions, "миллиард", Gender.MASCULINE)
            parts.append(b_form)
            n %= 1_000_000_000

        # Миллионы
        if n >= 1_000_000:
            millions = n // 1_000_000
            parts.extend(self._group_to_words(millions, Gender.MASCULINE))
            m_form = self._noun_form_by_number(millions, "миллион", Gender.MASCULINE)
            parts.append(m_form)
            n %= 1_000_000

        # Тысячи (тысяча — женский род!)
        if n >= 1_000:
            thousands = n // 1_000
            parts.extend(self._group_to_words(thousands, Gender.FEMININE))
            t_form = self._noun_form_by_number(thousands, "тысяча", Gender.FEMININE)
            parts.append(t_form)
            n %= 1_000

        # Сотни, десятки, единицы
        if n > 0:
            parts.extend(self._group_to_words(n, gender))

        return parts

    def _group_to_words(self, n: int, gender: Gender) -> list[str]:
        """Перевести число 1..999 в слова с учётом рода (для 1 и 2)."""
        if n <= 0:
            return []
        result: list[str] = []

        hundreds = n // 100
        if hundreds > 0:
            result.append(_HUNDREDS[hundreds])

        remainder = n % 100

        if remainder == 0:
            pass
        elif remainder < 20:
            # Для 1 и 2 используем форму с родом
            if remainder == 1 and gender == Gender.FEMININE:
                result.append("одна")
            elif remainder == 2 and gender == Gender.FEMININE:
                result.append("две")
            else:
                result.append(_ONES[remainder])
        else:
            tens = remainder // 10
            ones = remainder % 10
            result.append(_TENS[tens])
            if ones > 0:
                if ones == 1 and gender == Gender.FEMININE:
                    result.append("одна")
                elif ones == 2 and gender == Gender.FEMININE:
                    result.append("две")
                else:
                    result.append(_ONES[ones])

        return result

    def _noun_form_by_number(self, n: int, word: str, gender: Gender) -> str:
        """
        Согласование существительного с числительным в именительном:
          1 → им.ед. (рубль), 2-4 → род.ед. (рубля), 5+ → род.мн. (рублей)
        """
        last_two = n % 100
        last_one = n % 10

        if 11 <= last_two <= 14:
            # «одиннадцать рублей» — всегда род.мн.
            return self._inflect_unit(word, Case.GENITIVE, Number.PLURAL)
        elif last_one == 1:
            return word  # именительный единственного
        elif 2 <= last_one <= 4:
            return self._inflect_unit(word, Case.GENITIVE, Number.SINGULAR)
        else:
            return self._inflect_unit(word, Case.GENITIVE, Number.PLURAL)

    # ── Склонение единицы измерения ──────────────────────────────

    def _inflect_unit_for_number(self, unit: str, number: int, target_case: Case) -> str:
        """
        Согласовать существительное с числительным в нужном падеже.

        В именительном/винительном: числительное управляет существительным.
        В косвенных падежах: существительное — множественное число того же падежа,
        КРОМЕ составных, заканчивающихся на «один» — единственное число.
        """
        if target_case in (Case.NOMINATIVE, Case.ACCUSATIVE):
            return self._noun_form_by_number(number, unit, Gender.MASCULINE)

        # Косвенные падежи: существительное во множественном числе того же падежа
        last_two = number % 100
        last_one = number % 10

        # Исключение: "двадцати одному рублю" (не "рублям")
        if last_one == 1 and last_two != 11:
            return self._inflect_unit(unit, target_case, Number.SINGULAR)
        else:
            return self._inflect_unit(unit, target_case, Number.PLURAL)

    def _inflect_unit(self, word: str, case: Case, number: Number) -> str:
        """Склонить существительное в указанный падеж и число через pymorphy3."""
        try:
            parses = self.morph.parse(word)
            if not parses:
                return word
            parse = parses[0]
            inflected = parse.inflect({case.value, number.value})
            if inflected is None:
                return word
            return inflected.word
        except Exception:
            return word

    # ── Склонение отдельного слова числительного ─────────────────

    def _inflect_word(self, word: str, case: Case) -> str:
        """Склонить одно слово числительного через pymorphy3."""
        try:
            parses = self.morph.parse(word)
            if not parses:
                return word
            # Предпочитаем NUMR-парсинг
            numr_parses = [p for p in parses if "NUMR" in p.tag]
            parse = numr_parses[0] if numr_parses else parses[0]
            inflected = parse.inflect({case.value})
            if inflected is None:
                return word
            return inflected.word
        except Exception:
            return word
