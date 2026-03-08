"""LRU-кэш для словоуровневого конвейера."""

from __future__ import annotations

import logging
import json
from collections import OrderedDict
from typing import Optional

from ..core.interfaces import ICacheBackend
from ..core.models import FullParadigm

logger = logging.getLogger(__name__)


class LRUCacheBackend(ICacheBackend):
    """L1 кэш: in-memory LRU на основе OrderedDict."""

    def __init__(self, max_size: int = 100_000):
        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._paradigm_cache: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "paradigm_size": len(self._paradigm_cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def get_paradigm(self, word: str) -> Optional[FullParadigm]:
        key = f"paradigm:{word.lower()}"
        if key in self._paradigm_cache:
            self._hits += 1
            self._paradigm_cache.move_to_end(key)
            try:
                return FullParadigm.model_validate_json(self._paradigm_cache[key])
            except Exception:
                del self._paradigm_cache[key]
                return None
        self._misses += 1
        return None

    def set_paradigm(self, word: str, paradigm: FullParadigm) -> None:
        key = f"paradigm:{word.lower()}"
        raw = paradigm.model_dump_json()
        if key in self._paradigm_cache:
            self._paradigm_cache.move_to_end(key)
        else:
            if len(self._paradigm_cache) >= self._max_size // 10:
                self._paradigm_cache.popitem(last=False)
        self._paradigm_cache[key] = raw

    def flush(self) -> None:
        self._cache.clear()
        self._paradigm_cache.clear()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(word: str, case: str, number: str) -> str:
        return f"{word.lower()}:{case}:{number}"
