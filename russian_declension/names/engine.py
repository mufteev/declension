"""
Модуль склонения русских имён (ФИО) — v2.

Исправление v1: суффикс «-ов» отрезался вместо сохранения.
  v1 (баг): «Иванов»[:-2] = «Иван» + «а» = «Ивана»  ✗
  v2 (fix): «Иванов» + «а» = «Иванова»                ✓

Правило для фамилий на -ов/-ев (мужских):
  stem = вся фамилия целиком, окончания: +а/+у/+а/+ым/+е
Для -ова/-ева (женских):
  stem = фамилия без последней «а», окончания: +ой/+ой/+у/+ой/+ой
"""

from __future__ import annotations
import re, logging
from typing import Optional
from enum import Enum
from ..core.enums import Case

logger = logging.getLogger(__name__)


class NameGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class NameEngine:
    def __init__(self):
        self._morph = None

    @property
    def morph(self):
        if self._morph is None:
            from pymorphy3 import MorphAnalyzer
            self._morph = MorphAnalyzer(lang="ru")
        return self._morph

    def inflect_name(self, full_name: str, target_case: Case,
                     gender: NameGender = NameGender.UNKNOWN) -> str:
        if target_case == Case.NOMINATIVE:
            return full_name
        parts = full_name.strip().split()
        if not parts:
            return full_name
        if gender == NameGender.UNKNOWN:
            gender = self._detect_gender(parts)

        if len(parts) == 1:
            return self._inflect_surname(parts[0], target_case, gender)
        if len(parts) == 2:
            if self._looks_like_surname(parts[0]):
                return (self._inflect_surname(parts[0], target_case, gender) + " " +
                        self._inflect_first_name(parts[1], target_case, gender))
            return (self._inflect_first_name(parts[0], target_case, gender) + " " +
                    self._inflect_surname(parts[1], target_case, gender))
        # 3+ слов: Фамилия Имя Отчество [...]
        s = self._inflect_surname(parts[0], target_case, gender)
        f = self._inflect_first_name(parts[1], target_case, gender)
        p = self._inflect_patronymic(parts[2], target_case, gender)
        rest = " ".join(parts[3:])
        return f"{s} {f} {p}" + (f" {rest}" if rest else "")

    # ══════════════════════════════════════════════════════════════
    # Фамилии — исправленная v2-логика
    # ══════════════════════════════════════════════════════════════

    def _inflect_surname(self, surname: str, case: Case, gender: NameGender) -> str:
        if case == Case.NOMINATIVE:
            return surname
        low = surname.lower()

        if "-" in surname:
            return "-".join(
                self._inflect_surname(p, case, gender) for p in surname.split("-")
            )
        if self._is_indeclinable(low, gender):
            return surname

        # -ский/-цкий/-ской/-цкой / -ская/-цкая
        if re.search(r"(ск|цк)(ий|ой|ая)$", low):
            return self._inflect_ski(surname, case, gender)

        # -ов/-ев/-ёв (мужская): stem = целая фамилия, append окончание
        if gender != NameGender.FEMALE and re.search(r"(ов|ев|ёв)$", low):
            return surname + {
                Case.GENITIVE: "а", Case.DATIVE: "у",
                Case.ACCUSATIVE: "а", Case.INSTRUMENTAL: "ым",
                Case.PREPOSITIONAL: "е",
            }.get(case, "")

        # -ова/-ева/-ёва (женская): stem = без «а»
        if gender == NameGender.FEMALE and re.search(r"(ов|ев|ёв)а$", low):
            stem = surname[:-1]   # Иванова → Иванов
            return stem + {
                Case.GENITIVE: "ой", Case.DATIVE: "ой",
                Case.ACCUSATIVE: "у", Case.INSTRUMENTAL: "ой",
                Case.PREPOSITIONAL: "ой",
            }.get(case, "а")

        # -ин/-ын (мужская)
        if gender != NameGender.FEMALE and re.search(r"(ин|ын)$", low):
            return surname + {
                Case.GENITIVE: "а", Case.DATIVE: "у",
                Case.ACCUSATIVE: "а", Case.INSTRUMENTAL: "ым",
                Case.PREPOSITIONAL: "е",
            }.get(case, "")

        # -ина/-ына (женская)
        if gender == NameGender.FEMALE and re.search(r"(ин|ын)а$", low):
            stem = surname[:-1]
            return stem + {
                Case.GENITIVE: "ой", Case.DATIVE: "ой",
                Case.ACCUSATIVE: "у", Case.INSTRUMENTAL: "ой",
                Case.PREPOSITIONAL: "ой",
            }.get(case, "а")

        # Женская на согласный → не склоняется
        if gender == NameGender.FEMALE and low[-1] not in "аяоеёуюиыьъ":
            return surname

        # Мужская на согласный → 2-е склонение
        if gender != NameGender.FEMALE and low[-1] not in "аяоеёуюиыьъ":
            return surname + {
                Case.GENITIVE: "а", Case.DATIVE: "у",
                Case.ACCUSATIVE: "а", Case.INSTRUMENTAL: "ом",
                Case.PREPOSITIONAL: "е",
            }.get(case, "")

        return self._inflect_via_pymorphy(surname, case)

    def _inflect_ski(self, surname: str, case: Case, gender: NameGender) -> str:
        low = surname.lower()
        if low.endswith("ая"):
            stem = surname[:-2]
            return stem + {Case.GENITIVE: "ой", Case.DATIVE: "ой",
                           Case.ACCUSATIVE: "ую", Case.INSTRUMENTAL: "ой",
                           Case.PREPOSITIONAL: "ой"}.get(case, "ая")
        if gender == NameGender.FEMALE:
            stem = surname[:-2]
            return stem + {Case.NOMINATIVE: "ая", Case.GENITIVE: "ой",
                           Case.DATIVE: "ой", Case.ACCUSATIVE: "ую",
                           Case.INSTRUMENTAL: "ой", Case.PREPOSITIONAL: "ой"}.get(case, "ая")
        stem = surname[:-2]
        return stem + {Case.GENITIVE: "ого", Case.DATIVE: "ому",
                       Case.ACCUSATIVE: "ого", Case.INSTRUMENTAL: "им",
                       Case.PREPOSITIONAL: "ом"}.get(case, "ий")

    def _is_indeclinable(self, low: str, gender: NameGender) -> bool:
        if re.search(r"(ых|их)$", low): return True
        if low.endswith("ко") and len(low) > 3: return True
        if low.endswith("дзе") or low.endswith("швили"): return True
        if low[-1] in "оуюэие" and len(low) > 2: return True
        return False

    def _looks_like_surname(self, word: str) -> bool:
        low = word.lower()
        return any(re.search(p, low) for p in [
            r"(ов|ев|ёв)$", r"(ов|ев|ёв)а$", r"(ин|ын)$", r"(ин|ын)а$",
            r"(ск|цк)(ий|ая|ой)$", r"(ых|их)$", r"ко$"])

    # ── Имена ────────────────────────────────────────────────────
    def _inflect_first_name(self, name: str, case: Case, gender: NameGender) -> str:
        return self._inflect_via_pymorphy(name, case)

    # ── Отчества ─────────────────────────────────────────────────
    def _inflect_patronymic(self, pat: str, case: Case, gender: NameGender) -> str:
        low = pat.lower()
        for suf in ("ович", "евич"):
            if low.endswith(suf):
                stem = pat[:-len(suf)]
                return stem + {Case.GENITIVE: suf+"а", Case.DATIVE: suf+"у",
                               Case.ACCUSATIVE: suf+"а", Case.INSTRUMENTAL: suf+"ем",
                               Case.PREPOSITIONAL: suf+"е"}.get(case, suf)
        if low.endswith("ич") and not low.endswith("вич"):
            stem = pat[:-2]
            return stem + {Case.GENITIVE: "ича", Case.DATIVE: "ичу",
                           Case.ACCUSATIVE: "ича", Case.INSTRUMENTAL: "ичом",
                           Case.PREPOSITIONAL: "иче"}.get(case, "ич")
        for suf in ("овна", "евна"):
            if low.endswith(suf):
                stem = pat[:-len(suf)]
                s = suf[:-1]
                return stem + {Case.GENITIVE: s+"ы", Case.DATIVE: s+"е",
                               Case.ACCUSATIVE: s+"у", Case.INSTRUMENTAL: s+"ой",
                               Case.PREPOSITIONAL: s+"е"}.get(case, suf)
        for suf in ("инична", "ична"):
            if low.endswith(suf):
                stem = pat[:-len(suf)]
                s = suf[:-1]
                return stem + {Case.GENITIVE: s+"ы", Case.DATIVE: s+"е",
                               Case.ACCUSATIVE: s+"у", Case.INSTRUMENTAL: s+"ой",
                               Case.PREPOSITIONAL: s+"е"}.get(case, suf)
        return self._inflect_via_pymorphy(pat, case)

    # ── Пол ──────────────────────────────────────────────────────
    def _detect_gender(self, parts: list[str]) -> NameGender:
        for p in parts:
            low = p.lower()
            if low.endswith(("ович","евич")): return NameGender.MALE
            if low.endswith(("овна","евна","ична","инична")): return NameGender.FEMALE
        for p in parts:
            low = p.lower()
            if re.search(r"(ов|ев|ёв)а$", low): return NameGender.FEMALE
            if re.search(r"(ов|ев|ёв)$", low): return NameGender.MALE
            if low.endswith(("ская","цкая")): return NameGender.FEMALE
            if low.endswith(("ский","цкий")): return NameGender.MALE
        return NameGender.UNKNOWN

    def _inflect_via_pymorphy(self, word: str, case: Case) -> str:
        try:
            parses = self.morph.parse(word)
            if not parses: return word
            inflected = parses[0].inflect({case.value})
            if inflected is None: return word
            r = inflected.word
            if word[0].isupper() and r: r = r[0].upper() + r[1:]
            return r
        except Exception:
            return word
