"""
DeclensionService v2 — центральный оркестратор с GPU-поддержкой.

GPU-компоненты РЕАЛЬНО применяются в конвейере:

  AnimacyClassifier.predict()
    → в _inflect_word(), когда target_case=ACCUSATIVE и слово OOV (confidence < 0.6)
    → предсказывает одушевлённость, чтобы правильно выбрать Вин=Род или Вин=Им
    → вызов: pymorphy.inflect_with_agreement(word, case, animacy=predicted)

  MetaEnsemble.select_best()
    → в _inflect_word(), когда в FallbackChain 2+ engine'а
    → вместо каскадного fallback собирает результаты от ВСЕХ engine'ов
      через chain.inflect_all() и даёт MetaEnsemble выбрать лучший
    → полезен когда pymorphy ошибается, а ruT5 прав (или наоборот)

  BertValidator.validate_inflection_result()
    → в _inflect_word() для low-confidence результатов (confidence < 0.8)
    → в _inflect_phrase() для постпроверки склонения головного слова
    → снижает confidence если BERT-тегер видит расхождение с целевым падежом
"""

from __future__ import annotations
import re, time, logging
from typing import Optional
from enum import Enum

from .core.enums import Case, Number, Gender, Animacy
from .core.models import InflectionResult, FullParadigm
from .engines.pymorphy_engine import PymorphyEngine
from .engines.cache import LRUCacheBackend
from .engines.fallback_chain import FallbackChain
from .names.engine import NameEngine, NameGender
from .numerals.engine import NumeralEngine
from .organizations.engine import OrganizationEngine
from .phrases.engine import PhraseEngine

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    AUTO = "auto"
    WORD = "word"
    NAME = "name"
    ORGANIZATION = "org"
    NUMERAL = "numeral"
    PHRASE = "phrase"


class DeclensionService:
    """
    Фасад системы склонения v2 — CPU + опциональный GPU.

    GPU-компоненты загружаются только если указаны пути к моделям.
    Без GPU система работает точно так же, как План 1.
    """

    def __init__(self, cache_size: int = 100_000,
                 rut5_model_path: Optional[str] = None,
                 bert_validator_model: Optional[str] = None,
                 animacy_model_path: Optional[str] = None,
                 ensemble_model_path: Optional[str] = None,
                 gpu_device: str = "auto"):

        # ── Фаза 1: CPU-конвейер ─────────────────────────────────
        self._pymorphy = PymorphyEngine()
        self._cache = LRUCacheBackend(max_size=cache_size)

        # Собираем список engine'ов для FallbackChain
        engines = [self._pymorphy]

        # ── GPU: ruT5 engine (Фаза 2 Плана 2) ────────────────────
        self._rut5 = None
        if rut5_model_path:
            try:
                from .gpu.rut5_engine import RuT5Engine
                self._rut5 = RuT5Engine(model_path=rut5_model_path, device=gpu_device)
                if self._rut5.is_available:
                    engines.append(self._rut5)
                    logger.info("GPU: RuT5Engine активирован.")
            except Exception as exc:
                logger.warning("GPU: RuT5Engine недоступен: %s", exc)

        self._chain = FallbackChain(engines=engines, cache=self._cache)

        # ── Фаза 2: Именованные сущности (CPU) ───────────────────
        self._name_engine = NameEngine()
        self._numeral_engine = NumeralEngine()
        self._org_engine = OrganizationEngine()

        # ── Фаза 3: Фразовый движок (CPU + опц. GPU parser) ─────
        self._phrase_engine = PhraseEngine()

        # ── GPU: Валидатор (Фаза 3 Плана 2) ──────────────────────
        self._bert_validator = None
        if bert_validator_model:
            try:
                from .gpu.bert_validator import BertValidator
                self._bert_validator = BertValidator(
                    model_path=bert_validator_model, device=gpu_device)
                if self._bert_validator.is_available:
                    logger.info("GPU: BertValidator активирован.")
            except Exception as exc:
                logger.warning("GPU: BertValidator недоступен: %s", exc)

        # ── GPU: Классификатор одушевлённости (Фаза 4 Плана 2) ───
        self._animacy_clf = None
        if animacy_model_path:
            try:
                from .gpu.animacy_classifier import AnimacyClassifier
                self._animacy_clf = AnimacyClassifier(
                    model_path=animacy_model_path, device=gpu_device)
                if self._animacy_clf.is_available:
                    logger.info("GPU: AnimacyClassifier активирован.")
            except Exception as exc:
                logger.warning("GPU: AnimacyClassifier недоступен: %s", exc)

        # ── GPU: Мета-ансамбль (Фаза 5 Плана 2) ─────────────────
        self._ensemble = None
        if ensemble_model_path:
            try:
                from .gpu.ensemble import MetaEnsemble
                self._ensemble = MetaEnsemble(
                    model_path=ensemble_model_path)
                if self._ensemble.is_available:
                    logger.info("GPU: MetaEnsemble активирован.")
            except Exception as exc:
                logger.warning("GPU: MetaEnsemble недоступен: %s", exc)

        # ── Кэш фраз ─────────────────────────────────────────────
        self._phrase_cache: dict[str, str] = {}

        gpu_status = []
        if self._rut5 and self._rut5.is_available: gpu_status.append("RuT5")
        if self._bert_validator and self._bert_validator.is_available: gpu_status.append("BertValidator")
        if self._animacy_clf and self._animacy_clf.is_available: gpu_status.append("AnimacyCLF")
        if self._ensemble and self._ensemble.is_available: gpu_status.append("Ensemble")
        gpu_info = ", ".join(gpu_status) if gpu_status else "нет (CPU-only)"
        logger.info("DeclensionService v2. GPU: %s", gpu_info)

    # ══════════════════════════════════════════════════════════════
    # Главный метод
    # ══════════════════════════════════════════════════════════════

    def inflect(self, text: str, target_case: Case,
                entity_type: EntityType = EntityType.AUTO,
                target_number: Optional[Number] = None,
                gender: Optional[str] = None,
                context: Optional[str] = None) -> dict:
        t0 = time.perf_counter()

        if target_case == Case.NOMINATIVE:
            return self._resp(text, text, 1.0, "passthrough", [], [])

        if entity_type == EntityType.AUTO:
            entity_type = self._detect_entity_type(text)

        cache_key = f"{text}|{target_case.value}|{entity_type.value}|{target_number}|{gender}"
        if cache_key in self._phrase_cache:
            result = self._resp(text, self._phrase_cache[cache_key], 1.0,
                              "phrase_cache", [], [],
                              elapsed_ms=(time.perf_counter()-t0)*1000)
            result['target_case'] = target_case
            result['target_case_readable'] = target_case.label
            return result

        try:
            if entity_type == EntityType.WORD:
                result = self._inflect_word(text, target_case, target_number, context)
            elif entity_type == EntityType.NAME:
                result = self._inflect_name(text, target_case, gender)
            elif entity_type == EntityType.ORGANIZATION:
                result = self._inflect_org(text, target_case)
            elif entity_type == EntityType.NUMERAL:
                result = self._inflect_numeral(text, target_case)
            else:
                result = self._inflect_phrase(text, target_case)
        except Exception as exc:
            logger.error("Ошибка '%s' [%s]: %s", text, entity_type, exc, exc_info=True)
            result = self._resp(text, text, 0.0, "error", [], [str(exc)])

        result['target_case'] = target_case
        result['target_case_readable'] = target_case.label

        self._phrase_cache[cache_key] = result["result"]
        result["elapsed_ms"] = round((time.perf_counter()-t0)*1000, 2)
        return result

    # ── Маршрутизация ────────────────────────────────────────────

    def _inflect_word(self, word, case, number=None, context=None):
        warnings = []

        # ────────────────────────────────────────────────────────
        # Шаг 1: AnimacyClassifier для винительного падежа OOV
        # ────────────────────────────────────────────────────────
        # Если целевой падеж — винительный, а слово — OOV (pymorphy не уверен
        # в одушевлённости), предсказываем одушевлённость нейросетью и
        # переинфлектируем с явным указанием anim/inan.
        if (case == Case.ACCUSATIVE
                and self._animacy_clf and self._animacy_clf.is_available):
            morph_info = self._pymorphy.analyze(word)
            # Условие: слово OOV (low score) или одушевлённость не определена
            if morph_info and (morph_info.score < 0.5 or morph_info.animacy is None):
                predicted_animacy, animacy_conf = self._animacy_clf.predict(word)
                logger.debug("AnimacyClassifier: '%s' → %s (conf=%.2f)",
                             word, predicted_animacy.value, animacy_conf)
                # Если классификатор уверен — переинфлектируем с явной одушевлённостью
                if animacy_conf > 0.6:
                    reinflected = self._pymorphy.inflect_with_agreement(
                        word, case,
                        gender=morph_info.gender,
                        number=number,
                        animacy=predicted_animacy,
                    )
                    if reinflected:
                        warnings.append(
                            f"animacy_predicted:{predicted_animacy.value}:{animacy_conf:.2f}")
                        return self._resp(
                            word, reinflected, 0.75,
                            "pymorphy+animacy_clf", [], warnings)

        # ────────────────────────────────────────────────────────
        # Шаг 2: MetaEnsemble — если 2+ engine'а, собрать все
        #         результаты и выбрать лучший
        # ────────────────────────────────────────────────────────
        if (self._ensemble and self._ensemble.is_available
                and self._chain.engine_count >= 2):
            all_results = self._chain.inflect_all(word, case, number, context)
            if len(all_results) >= 2:
                best = self._ensemble.select_best(all_results, word=word)
                logger.debug("MetaEnsemble: '%s' → selected '%s' from %s (conf=%.2f)",
                             word, best.engine,
                             [r.engine for r in all_results], best.confidence)
                ir = best
                ir.warnings.append(f"ensemble_selected_from:{len(all_results)}_engines")
            else:
                # Только один engine вернул результат — используем его
                ir = all_results[0] if all_results else self._chain.inflect(
                    word, case, number, context)
        else:
            # Нет ансамбля или один engine — стандартный каскад
            ir = self._chain.inflect(word, case, number, context)

        # ────────────────────────────────────────────────────────
        # Шаг 3: BertValidator — постпроверка low-confidence
        # ────────────────────────────────────────────────────────
        if (self._bert_validator and self._bert_validator.is_available
                and ir.confidence < 0.8):
            ir = self._bert_validator.validate_inflection_result(ir)

        ir.warnings.extend(warnings)
        return self._resp(word, ir.inflected_form, ir.confidence,
                          ir.engine, [], ir.warnings)

    def _inflect_name(self, name, case, gender=None):
        ng = {"male": NameGender.MALE, "female": NameGender.FEMALE}.get(
            gender, NameGender.UNKNOWN)
        result = self._name_engine.inflect_name(name, case, ng)
        return self._resp(name, result, 0.9 if result != name else 0.5,
                          "name_engine", [], [])

    def _inflect_org(self, org, case):
        result = self._org_engine.inflect_org(org, case)
        return self._resp(org, result, 0.95 if result != org else 0.8,
                          "org_engine", [], [])

    def _inflect_numeral(self, text, case):
        match = re.match(r'^(\d+)\s*(.*)$', text.strip())
        if match:
            number = int(match.group(1))
            unit = match.group(2).strip() or None
            result = self._numeral_engine.inflect_numeral(number, case, unit=unit)
            return self._resp(text, result, 0.95, "numeral_engine", [], [])
        return self._inflect_phrase(text, case)

    def _inflect_phrase(self, phrase, case):
        result_text = self._phrase_engine.inflect_phrase(phrase, case)
        conf = 0.85 if result_text != phrase else 0.5
        warnings = []

        # BertValidator: проверяем склонение головного слова фразы
        if (self._bert_validator and self._bert_validator.is_available
                and conf < 0.9):
            # Извлекаем первое изменившееся слово (предполагаемая голова)
            orig_words = phrase.split()
            result_words = result_text.split()
            for ow, rw in zip(orig_words, result_words):
                if ow != rw:
                    # Это слово изменилось — проверяем его
                    validation = self._bert_validator.validate(
                        word=rw, expected_case=case,
                        context=f"Это {result_text}."
                    )

                    if not validation["valid"]:
                        conf *= 0.6
                        warnings.append(
                            f"bert_phrase_mismatch:word={rw},"
                            f"expected={case.value},"
                            f"predicted={validation['predicted_case']}")
                    else:
                        # BERT подтвердил — повышаем уверенность
                        conf = min(conf * 1.1, 0.95)
                        warnings.append("bert_phrase_validated")
                    break  # Проверяем только голову

        return self._resp(phrase, result_text, conf, "phrase_engine", [], warnings)

    # ── Автоопределение типа ─────────────────────────────────────

    def _detect_entity_type(self, text: str) -> EntityType:
        stripped = text.strip()
        if re.match(r'^\d', stripped):
            return EntityType.NUMERAL
        if OrganizationEngine.is_organization(stripped):
            return EntityType.ORGANIZATION
        words = stripped.split()
        if len(words) == 1:
            return EntityType.WORD
        if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
            if any(w.lower().endswith(("ович","евич","ич","овна","евна","ична"))
                   for w in words):
                return EntityType.NAME
            if any(re.search(r'(ов|ев|ёв|ин|ын|ский|цкий|ова|ева|ина|ына|ская|цкая)$',
                             w.lower()) for w in words):
                return EntityType.NAME
        return EntityType.PHRASE

    # ── Парадигма ────────────────────────────────────────────────

    def paradigm(self, word: str) -> Optional[dict]:
        p = self._chain.paradigm(word)
        return p.model_dump() if p else None

    # ── Пакет ────────────────────────────────────────────────────

    def inflect_batch(self, items: list[dict]) -> list[dict]:
        return [
            self.inflect(
                text=it["text"], target_case=Case(it["target_case"]),
                entity_type=EntityType(it.get("entity_type", "auto")),
                target_number=Number(it["target_number"]) if it.get("target_number") else None,
                gender=it.get("gender"), context=it.get("context"))
            for it in items
        ]

    # ── Health ───────────────────────────────────────────────────

    def health(self) -> dict:
        ch = self._chain.healthcheck()
        gpu_components = {}
        if self._rut5:
            gpu_components["rut5"] = "active" if self._rut5.is_available else "inactive"
        if self._bert_validator:
            gpu_components["bert_validator"] = "active" if self._bert_validator.is_available else "inactive"
        if self._animacy_clf:
            gpu_components["animacy_clf"] = "active" if self._animacy_clf.is_available else "inactive"
        if self._ensemble:
            gpu_components["ensemble"] = "active" if self._ensemble.is_available else "inactive"

        return {
            "status": ch["status"], "version": "0.4.0",
            "engines": ch["engines"], "metrics": ch["metrics"],
            "gpu_components": gpu_components or "none (CPU-only)",
            "phrase_cache_size": len(self._phrase_cache),
        }

    @staticmethod
    def _resp(original, result, confidence, engine, details, warnings,
              elapsed_ms=0.0):
        return {"original": original, "result": result,
                "confidence": round(confidence, 4), "engine": engine,
                "details": details, "warnings": warnings,
                "elapsed_ms": round(elapsed_ms, 2)}
