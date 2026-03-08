"""Абстрактные интерфейсы — контракт для всех engine'ов склонения."""

from abc import ABC, abstractmethod
from typing import Optional

from .enums import Case, Number
from .models import InflectionResult, FullParadigm, MorphInfo


class IDeclensionEngine(ABC):
    """Базовый интерфейс для любого engine склонения."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def confidence_threshold(self) -> float:
        return 0.5

    @abstractmethod
    def inflect(
        self, word: str, target_case: Case,
        target_number: Optional[Number] = None,
        context: Optional[str] = None,
    ) -> Optional[InflectionResult]:
        ...

    @abstractmethod
    def analyze(self, word: str) -> Optional[MorphInfo]:
        ...

    @abstractmethod
    def paradigm(self, word: str) -> Optional[FullParadigm]:
        ...

    def healthcheck(self) -> bool:
        return True


class ICacheBackend(ABC):
    """Абстракция кэша — LRU / Redis."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def get_paradigm(self, word: str) -> Optional[FullParadigm]:
        ...

    @abstractmethod
    def set_paradigm(self, word: str, paradigm: FullParadigm) -> None:
        ...
