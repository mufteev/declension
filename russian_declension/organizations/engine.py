"""
Модуль склонения названий организаций.

Ключевое правило (кавычечное правило):
  При наличии родового слова (ООО, ОАО, театр, фирма) закавыченное
  название НЕ склоняется: «правление ООО «Ромашка»», не «Ромашки».
  Без родового слова название склоняется: «работать в «Газпроме»».

Аббревиатуры юридических форм (ООО, ЗАО, ПАО) не склоняются.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from ..core.enums import Case

logger = logging.getLogger(__name__)

# Аббревиатуры организационно-правовых форм (несклоняемые)
_LEGAL_ABBREVS = {
    "ООО", "ОАО", "ЗАО", "ПАО", "АО", "НАО",
    "ИП", "ГУП", "МУП", "ФГУП", "НКО", "АНО",
    "СРО", "ТСЖ", "ТСН", "ПБОЮЛ", "КФХ",
}

# Родовые слова, после которых кавычечное название не склоняется
_GENERIC_WORDS = {
    "общество", "товарищество", "предприятие", "организация",
    "учреждение", "компания", "фирма", "корпорация", "концерн",
    "холдинг", "группа", "банк", "фонд", "театр", "музей",
    "издательство", "агентство", "бюро", "завод", "фабрика",
    "комбинат", "институт", "университет", "академия", "школа",
    "больница", "клиника", "аптека", "магазин", "ресторан",
    "кафе", "отель", "гостиница", "санаторий", "клуб",
}

# Паттерн для обнаружения кавычечных конструкций
# Поддерживаем «», "", '' и просто кавычки
_QUOTE_PATTERN = re.compile(
    r'([«""\'])(.*?)([»""\'])'
)


class OrganizationEngine:
    """
    Движок склонения названий организаций.

    Использование:
        engine = OrganizationEngine()
        result = engine.inflect_org('ООО «Ромашка»', Case.GENITIVE)
        # → 'ООО «Ромашка»' (кавычечное правило: не склоняем)

        result = engine.inflect_org('«Газпром»', Case.PREPOSITIONAL)
        # → '«Газпроме»' (без родового слова — склоняем)
    """

    def __init__(self):
        self._morph = None

    @property
    def morph(self):
        if self._morph is None:
            from pymorphy3 import MorphAnalyzer
            self._morph = MorphAnalyzer(lang="ru")
        return self._morph

    def inflect_org(self, org_name: str, target_case: Case) -> str:
        """
        Склонить название организации в целевой падеж.

        Алгоритм:
          1. Разделить на компоненты: аббревиатура + родовое слово + кавычечное имя
          2. Аббревиатуры — не склоняем
          3. Если есть родовое слово или аббревиатура — кавычечное имя не склоняем
          4. Родовое слово склоняем
          5. Если нет родового слова и аббревиатуры — кавычечное имя склоняем
        """
        if target_case == Case.NOMINATIVE:
            return org_name

        # Парсинг структуры
        parsed = self._parse_org_name(org_name)

        if parsed is None:
            # Нет кавычек — пробуем склонить как обычную фразу
            return self._inflect_plain_org(org_name, target_case)

        abbrev, generic, open_q, name, close_q, rest = parsed

        result_parts = []

        # Аббревиатура — без изменений
        if abbrev:
            result_parts.append(abbrev)

        # Родовое слово — склоняем
        if generic:
            inflected_generic = self._inflect_word(generic, target_case)
            result_parts.append(inflected_generic)

        # Кавычечное название
        has_parent = bool(abbrev or generic)
        if has_parent:
            # Кавычечное правило: название остаётся в именительном
            result_parts.append(f"{open_q}{name}{close_q}")
        else:
            # Без родового слова — склоняем название внутри кавычек
            inflected_name = self._inflect_word(name, target_case)
            result_parts.append(f"{open_q}{inflected_name}{close_q}")

        if rest:
            result_parts.append(rest)

        return " ".join(result_parts)

    # ── Парсинг структуры названия организации ───────────────────

    def _parse_org_name(self, text: str) -> Optional[tuple]:
        """
        Разобрать название на компоненты:
          (аббревиатура, родовое_слово, открывающая_кавычка, название, закрывающая_кавычка, остаток)

        Возвращает None, если нет кавычек.
        """
        match = _QUOTE_PATTERN.search(text)
        if not match:
            return None

        before = text[:match.start()].strip()
        open_q = match.group(1)
        name = match.group(2)
        close_q = match.group(3)
        after = text[match.end():].strip()

        # Разделяем before на аббревиатуру и родовое слово
        abbrev = ""
        generic = ""

        if before:
            words = before.split()
            remaining = []
            for w in words:
                if w.upper() in _LEGAL_ABBREVS:
                    abbrev = (abbrev + " " + w).strip() if abbrev else w
                elif w.lower() in _GENERIC_WORDS:
                    generic = (generic + " " + w).strip() if generic else w
                else:
                    remaining.append(w)
            if remaining:
                generic = (generic + " " + " ".join(remaining)).strip() if generic else " ".join(remaining)

        return abbrev, generic, open_q, name, close_q, after

    def _inflect_plain_org(self, name: str, case: Case) -> str:
        """Склонить название без кавычек (одно слово или простая фраза)."""
        words = name.split()
        if len(words) == 1:
            return self._inflect_word(words[0], case)
        # Для многословных: склоняем всё (упрощённо — склоняем каждое слово)
        return " ".join(self._inflect_word(w, case) for w in words)

    def _inflect_word(self, word: str, case: Case) -> str:
        """Склонить одно слово через pymorphy3."""
        # Проверяем, не аббревиатура ли
        if word.upper() in _LEGAL_ABBREVS or word.isupper():
            return word  # Аббревиатуры не склоняем

        try:
            parses = self.morph.parse(word)
            if not parses:
                return word
            parse = parses[0]
            inflected = parse.inflect({case.value})
            if inflected is None:
                return word
            result = inflected.word
            if word[0].isupper() and result:
                result = result[0].upper() + result[1:]
            return result
        except Exception:
            return word

    @staticmethod
    def is_organization(text: str) -> bool:
        """Эвристика: является ли текст названием организации."""
        upper_text = text.upper()
        # Содержит аббревиатуру юрлица
        if any(abbr in upper_text.split() for abbr in _LEGAL_ABBREVS):
            return True
        # Содержит кавычки и перед ними есть родовое слово
        if _QUOTE_PATTERN.search(text):
            before = text[:_QUOTE_PATTERN.search(text).start()].strip().lower()
            if any(gw in before for gw in _GENERIC_WORDS):
                return True
        return False
