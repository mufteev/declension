"""
Доменные модели системы склонения.

Pydantic-модели для валидации, сериализации и единого контракта
между всеми engine'ами системы.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from .enums import Case, Gender, Number, Animacy, SpecialGroup


class MorphInfo(BaseModel):
    """Морфологическая характеристика слова."""

    lemma: str = Field(..., description="Лемма (начальная форма)")
    gender: Optional[Gender] = Field(None, description="Род")
    animacy: Optional[Animacy] = Field(None, description="Одушевлённость")
    number: Optional[Number] = Field(None, description="Число исходной формы")
    case: Optional[Case] = Field(None, description="Падеж исходной формы")
    special_groups: list[SpecialGroup] = Field(
        default_factory=list,
        description="Особые морфологические группы (Fixd, Pltm и т.д.)",
    )
    is_noun: bool = Field(True, description="Является ли слово существительным")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность анализа")

    @property
    def is_indeclinable(self) -> bool:
        return SpecialGroup.INDECLINABLE in self.special_groups

    @property
    def is_pluralia_tantum(self) -> bool:
        return SpecialGroup.PLURALIA_TANTUM in self.special_groups

    @property
    def is_proper_noun(self) -> bool:
        return any(
            g in self.special_groups
            for g in (SpecialGroup.PROPER_NAME, SpecialGroup.SURNAME,
                      SpecialGroup.TOPONYM, SpecialGroup.PATRONYMIC)
        )


class InflectionResult(BaseModel):
    """Результат склонения одного слова — центральный контракт системы."""

    word: str = Field(..., description="Исходное слово")
    inflected_form: str = Field(..., description="Склонённая форма")
    target_case: Case = Field(..., description="Целевой падеж")
    target_number: Number = Field(Number.SINGULAR, description="Целевое число")
    morph_info: Optional[MorphInfo] = Field(None, description="Морфологическая информация")
    engine: str = Field("unknown", description="Имя engine, сгенерировавшего результат")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность генерации")
    is_fallback: bool = Field(False, description="True если сработал не первый engine")
    warnings: list[str] = Field(default_factory=list, description="Предупреждения")


class FullParadigm(BaseModel):
    """Полная парадигма склонения — все падежные формы слова."""

    word: str
    morph_info: MorphInfo
    forms: dict[str, Optional[str]] = Field(default_factory=dict)
    engine: str = "unknown"

    def get_form(self, case: Case, number: Number) -> Optional[str]:
        key = f"{case.value}_{number.value}"
        return self.forms.get(key)

    @staticmethod
    def make_key(case: Case, number: Number) -> str:
        return f"{case.value}_{number.value}"


class InflectionRequest(BaseModel):
    """Запрос на склонение — входная модель API."""

    word: str = Field(..., min_length=1, max_length=100, description="Слово для склонения")
    target_case: Case = Field(..., description="Целевой падеж")
    target_number: Optional[Number] = Field(None, description="Целевое число (None = сохранить)")
    context: Optional[str] = Field(None, max_length=500, description="Контекст предложения")


class BatchInflectionRequest(BaseModel):
    """Пакетный запрос на склонение."""

    items: list[InflectionRequest] = Field(..., min_length=1, max_length=1000)


class ParadigmRequest(BaseModel):
    """Запрос на получение полной парадигмы слова."""

    word: str = Field(..., min_length=1, max_length=100, description="Слово")
