"""
Грамматические категории русского языка.

Все перечисления содержат:
  - value: код pymorphy3 (граммема OpenCorpora)
  - label: человекочитаемое название на русском
  - mapping на pymorphy3 grammemes для бесшовной интеграции

Это «единый язык» (ubiquitous language) всей системы.
Нейросетевой fallback тоже будет оперировать этими enum'ами —
он получает Case + Number + Gender, а возвращает строку-словоформу.
"""

from enum import Enum


class Case(str, Enum):
    """Шесть падежей русского языка + два дополнительных (местный, партитивный)."""

    NOMINATIVE = "nomn"       # Именительный   — кто? что?
    GENITIVE = "gent"         # Родительный     — кого? чего?
    DATIVE = "datv"           # Дательный       — кому? чему?
    ACCUSATIVE = "accs"       # Винительный     — кого? что?
    INSTRUMENTAL = "ablt"     # Творительный    — кем? чем?
    PREPOSITIONAL = "loct"    # Предложный      — о ком? о чём?
    # Дополнительные формы (pymorphy2 поддерживает)
    LOCATIVE = "loc2"         # Местный (второй предложный): в лесу, на мосту
    PARTITIVE = "gen2"        # Партитивный (второй родительный): чашка чаю

    @property
    def label(self) -> str:
        labels = {
            "nomn": "Именительный",
            "gent": "Родительный",
            "datv": "Дательный",
            "accs": "Винительный",
            "ablt": "Творительный",
            "loct": "Предложный",
            "loc2": "Местный",
            "gen2": "Партитивный",
        }
        return labels[self.value]


class Gender(str, Enum):
    """Грамматический род."""

    MASCULINE = "masc"    # Мужской
    FEMININE = "femn"     # Женский
    NEUTER = "neut"       # Средний
    COMMON = "Ms-f"       # Общий род (сирота, коллега, плакса)

    @property
    def label(self) -> str:
        labels = {
            "masc": "Мужской",
            "femn": "Женский",
            "neut": "Средний",
            "Ms-f": "Общий",
        }
        return labels[self.value]


class Number(str, Enum):
    """Грамматическое число."""

    SINGULAR = "sing"     # Единственное
    PLURAL = "plur"       # Множественное

    @property
    def label(self) -> str:
        return "Единственное" if self.value == "sing" else "Множественное"


class Animacy(str, Enum):
    """Одушевлённость — критична для винительного падежа."""

    ANIMATE = "anim"      # Одушевлённое (Вин. = Род.)
    INANIMATE = "inan"    # Неодушевлённое (Вин. = Им.)

    @property
    def label(self) -> str:
        return "Одушевлённое" if self.value == "anim" else "Неодушевлённое"


class SpecialGroup(str, Enum):
    """
    Специальные морфологические группы — маркеры исключений.
    Используются для быстрого определения стратегии обработки.
    """

    INDECLINABLE = "Fixd"     # Несклоняемое (метро, кафе, такси)
    PLURALIA_TANTUM = "Pltm"  # Только множественное (ножницы, брюки)
    SINGULARIA_TANTUM = "Sgtm"  # Только единственное (молоко, золото)
    SUBSTANTIVIZED = "Subx"   # Субстантивированное прилагательное (столовая)
    PROPER_NAME = "Name"      # Имя собственное
    SURNAME = "Surn"          # Фамилия
    PATRONYMIC = "Patr"       # Отчество
    TOPONYM = "Geox"          # Топоним
