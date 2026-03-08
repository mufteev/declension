"""FallbackChain — оркестратор «Кэш → Словарь → Нейросеть»."""

from __future__ import annotations

import time
import logging
from typing import Optional
from collections import defaultdict

from ..core.enums import Case, Number
from ..core.models import InflectionResult, FullParadigm, MorphInfo
from ..core.interfaces import IDeclensionEngine, ICacheBackend
from .cache import LRUCacheBackend

logger = logging.getLogger(__name__)


class FallbackChain:
    """
    Оркестратор fallback-цепочки склонения.
    ВСЕГДА возвращает InflectionResult — никогда None.
    """

    def __init__(
        self,
        engines: list[IDeclensionEngine],
        cache: Optional[ICacheBackend] = None,
    ):
        if not engines:
            raise ValueError("FallbackChain требует хотя бы один engine.")
        self._engines = engines
        self._cache = cache or LRUCacheBackend()
        self._metrics: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "successes": 0, "total_ms": 0.0}
        )
        self._metrics["cache"] = {"calls": 0, "successes": 0, "total_ms": 0.0}

    def inflect(
        self, word: str, target_case: Case,
        target_number: Optional[Number] = None,
        context: Optional[str] = None,
    ) -> InflectionResult:
        effective_number = target_number or Number.SINGULAR

        # L1: кэш
        cache_key = LRUCacheBackend.make_key(word, target_case.value, effective_number.value)
        t0 = time.perf_counter()
        cached = self._cache.get(cache_key)
        cache_ms = (time.perf_counter() - t0) * 1000
        self._metrics["cache"]["calls"] += 1
        self._metrics["cache"]["total_ms"] += cache_ms

        if cached is not None:
            self._metrics["cache"]["successes"] += 1
            return InflectionResult(
                word=word, inflected_form=cached,
                target_case=target_case, target_number=effective_number,
                engine="cache", confidence=1.0,
            )

        # Перебор engine'ов
        best_result: Optional[InflectionResult] = None
        is_fallback = False

        for engine in self._engines:
            t0 = time.perf_counter()
            try:
                result = engine.inflect(word, target_case, target_number, context)
            except Exception as exc:
                logger.error("Engine '%s' error for '%s': %s", engine.name, word, exc)
                self._metrics[engine.name]["calls"] += 1
                is_fallback = True
                continue

            engine_ms = (time.perf_counter() - t0) * 1000
            self._metrics[engine.name]["calls"] += 1
            self._metrics[engine.name]["total_ms"] += engine_ms

            if result is None:
                is_fallback = True
                continue

            result.is_fallback = is_fallback

            if result.confidence >= engine.confidence_threshold:
                self._metrics[engine.name]["successes"] += 1
                self._cache.set(cache_key, result.inflected_form)
                return result

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result
            is_fallback = True

        if best_result is not None:
            best_result.warnings.append("low_confidence_best_effort")
            self._cache.set(cache_key, best_result.inflected_form)
            return best_result

        return InflectionResult(
            word=word, inflected_form=word,
            target_case=target_case, target_number=effective_number,
            engine="none", confidence=0.0,
            is_fallback=True, warnings=["all_engines_failed"],
        )

    def analyze(self, word: str) -> Optional[MorphInfo]:
        for engine in self._engines:
            try:
                result = engine.analyze(word)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def paradigm(self, word: str) -> Optional[FullParadigm]:
        cached = self._cache.get_paradigm(word)
        if cached is not None:
            return cached
        for engine in self._engines:
            try:
                result = engine.paradigm(word)
                if result is not None:
                    self._cache.set_paradigm(word, result)
                    return result
            except Exception:
                continue
        return None

    def inflect_batch(self, items: list[tuple[str, Case, Optional[Number]]]) -> list[InflectionResult]:
        return [self.inflect(w, c, n) for w, c, n in items]

    @property
    def metrics(self) -> dict:
        result = {}
        for name, data in self._metrics.items():
            calls = data["calls"]
            result[name] = {
                "calls": int(calls),
                "successes": int(data["successes"]),
                "hit_rate": round(data["successes"] / calls, 4) if calls > 0 else 0.0,
                "avg_ms": round(data["total_ms"] / calls, 3) if calls > 0 else 0.0,
            }
        if hasattr(self._cache, "stats"):
            result["cache_storage"] = self._cache.stats
        return result

    def healthcheck(self) -> dict:
        status = {}
        all_healthy = False
        for engine in self._engines:
            try:
                healthy = engine.healthcheck()
                status[engine.name] = "ok" if healthy else "degraded"
                if healthy:
                    all_healthy = True
            except Exception as exc:
                status[engine.name] = f"error: {exc}"
        return {"status": "ok" if all_healthy else "degraded", "engines": status, "metrics": self.metrics}
