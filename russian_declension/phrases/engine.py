"""
Фразовый движок склонения — v3.

Исправления v3 (на основе 560-тестовой выборки, 20 ошибок):

  P0 (10 ошибок): _reassemble() заменён на позиционную сборку.
     Старый str.replace() промахивался при совпадении подстрок или
     различиях ё/е. Новый метод сопоставляет токены словам по ИНДЕКСУ.
     Добавлена нормализация ё↔е при вызове pymorphy.

  P1 (5 ошибок): Топонимы при классификаторе «Республика/Область/Край».
     «Республика Татарстан» → «Республики Татарстан» (Татарстан НЕ склоняется).
     Правило: если голова — топоним-классификатор, а зависимое с заглавной буквы
     оканчивается на согласный — это нерусский/тюркский топоним, не склоняем.

  P3 (2 ошибки): «Восьмой том» — порядковое числительное.
     Добавлены NUMR, ANUM, ADJF в список модификаторов, которые
     распознаются ДО нахождения головного существительного.
"""

from __future__ import annotations
import re, logging
from typing import Optional
from dataclasses import dataclass, field
from ..core.enums import Case, Gender, Number, Animacy

logger = logging.getLogger(__name__)

_NATASHA_AVAILABLE = False
try:
    from natasha import (Segmenter, MorphVocab, NewsEmbedding,
                         NewsMorphTagger, NewsSyntaxParser, Doc)
    _NATASHA_AVAILABLE = True
except ImportError:
    pass

_PREPOSITIONS = {
    "с", "в", "на", "за", "о", "об", "обо", "по", "к", "ко", "у", "из",
    "от", "до", "для", "при", "про", "без", "над", "под", "между",
    "через", "перед", "после", "около", "среди", "вместо", "ради",
}

# ── Топонимные классификаторы (P1) ───────────────────────────────
# При наличии классификатора, имя собственное на согласный НЕ склоняется:
# «Республика Татарстан» → «Республики Татарстан» (не «Татарстана»)
_TOPONYM_CLASSIFIERS = {
    "республика", "область", "край", "округ", "район",
    "город", "село", "деревня", "посёлок", "поселок",
    "улица", "проспект", "переулок", "бульвар", "шоссе",
    "река", "озеро", "море", "гора", "остров", "мыс",
}

# POS-теги, которые являются модификаторами в именных группах
_MODIFIER_POS = {"ADJF", "PRTF", "ADJ", "NUMR", "ANUM"}


@dataclass
class TokenInfo:
    idx: int               # Позиция слова в оригинальной фразе (0-based)
    id: int                # ID для dependency tree
    text: str
    lemma: str = ""
    pos: str = ""
    dep_rel: str = ""
    head_id: int = 0
    feats: dict = field(default_factory=dict)
    inflected: str = ""
    should_inflect: bool = False


@dataclass
class PhraseAnalysis:
    tokens: list[TokenInfo]
    head_idx: int = -1
    head_gender: Optional[str] = None
    head_number: Optional[str] = None
    head_animacy: Optional[str] = None


def _normalize_yo(word: str) -> str:
    """ё → е для совместимости с pymorphy."""
    return word.replace("ё", "е").replace("Ё", "Е")


def _restore_yo(original: str, inflected: str) -> str:
    """
    Восстановить ё из оригинала в склонённую форму, если начало слова совпадает.
    «Расчётного» (из «Расчётный») → «Расчётного» (ё на позиции 5).
    """
    if "ё" not in original.lower():
        return inflected
    result = list(inflected)
    orig_lower = original.lower()
    infl_lower = inflected.lower()
    # Восстанавливаем ё в совпадающем префиксе
    min_len = min(len(orig_lower), len(infl_lower))
    for i in range(min_len):
        if orig_lower[i] == "ё" and infl_lower[i] == "е":
            result[i] = "ё" if original[i] == "ё" else "Ё"
    return "".join(result)


class PhraseEngine:
    def __init__(self):
        self._morph_engine = None
        self._segmenter = None
        self._morph_vocab = None
        self._emb = None
        self._morph_tagger = None
        self._syntax_parser = None
        self._natasha_ready = False
        self._init_attempted = False

    def _ensure_natasha(self) -> bool:
        if self._init_attempted:
            return self._natasha_ready
        self._init_attempted = True
        if not _NATASHA_AVAILABLE:
            logger.warning("Natasha не установлена — heuristic fallback.")
            return False
        try:
            self._segmenter = Segmenter()
            self._morph_vocab = MorphVocab()
            self._emb = NewsEmbedding()
            self._morph_tagger = NewsMorphTagger(self._emb)
            self._syntax_parser = NewsSyntaxParser(self._emb)
            self._natasha_ready = True
        except Exception as exc:
            logger.error("Natasha init error: %s", exc)
        return self._natasha_ready

    @property
    def morph_engine(self):
        if self._morph_engine is None:
            from ..engines.pymorphy_engine import PymorphyEngine
            self._morph_engine = PymorphyEngine()
        return self._morph_engine

    # ══════════════════════════════════════════════════════════════
    # Главный метод
    # ══════════════════════════════════════════════════════════════

    def inflect_phrase(self, phrase: str, target_case: Case) -> str:
        if target_case == Case.NOMINATIVE:
            return phrase
        phrase = phrase.strip()
        if not phrase:
            return phrase

        words = phrase.split()
        if len(words) == 1:
            return self._inflect_single_word(words[0], target_case)

        analysis = self._analyze_phrase(phrase)
        if analysis is None or analysis.head_idx < 0:
            return self._heuristic_inflect(phrase, target_case)

        head = analysis.tokens[analysis.head_idx]

        for i, token in enumerate(analysis.tokens):
            if not token.should_inflect:
                token.inflected = token.text
                continue

            if token.dep_rel == "root" or token.id == head.id:
                token.inflected = self._safe_inflect_word(
                    token.text, target_case)

            elif token.dep_rel in ("amod", "det"):
                token.inflected = self._inflect_agreement(
                    token.text, target_case,
                    analysis.head_gender, analysis.head_number,
                    analysis.head_animacy)

            elif token.dep_rel in ("flat:name", "flat", "nummod"):
                token.inflected = self._safe_inflect_word(
                    token.text, target_case)

            elif token.dep_rel == "appos":
                token.inflected = self._safe_inflect_word(
                    token.text, target_case)

            else:
                token.inflected = token.text

        return self._reassemble(phrase, analysis.tokens)

    def _inflect_single_word(self, word: str, case: Case) -> str:
        """Склонение одного слова с ё-нормализацией."""
        r = self.morph_engine.inflect(word, case)
        if r and r.inflected_form != word:
            return _restore_yo(word, r.inflected_form)
        # Попробуем с нормализованной ё
        normalized = _normalize_yo(word)
        if normalized != word:
            r = self.morph_engine.inflect(normalized, case)
            if r and r.inflected_form != normalized:
                return _restore_yo(word, r.inflected_form)
        return r.inflected_form if r else word

    def _safe_inflect_word(self, word: str, case: Case) -> str:
        """
        Склонение слова с ё-нормализацией и санитарной проверкой результата.
        Если pymorphy возвращает мусор — fallback на нормализованную версию.
        """
        # Попытка 1: оригинал
        r = self.morph_engine.inflect(word, case)
        if r:
            result = _restore_yo(word, r.inflected_form)
            if self._is_sane_inflection(word, result):
                return result

        # Попытка 2: с ё→е
        normalized = _normalize_yo(word)
        if normalized != word:
            r = self.morph_engine.inflect(normalized, case)
            if r:
                result = _restore_yo(word, r.inflected_form)
                if self._is_sane_inflection(word, result):
                    return result

        return word  # fallback — оригинал

    @staticmethod
    def _is_sane_inflection(original: str, inflected: str) -> bool:
        """Проверка адекватности результата: длина не должна сильно отличаться."""
        if not inflected:
            return False
        len_diff = abs(len(inflected) - len(original))
        # Обычное русское склонение меняет 1–3 символа
        if len_diff > 4:
            return False
        # Начало слова должно совпадать (хотя бы 60%)
        common_prefix = 0
        for a, b in zip(original.lower(), inflected.lower()):
            if a == b or (a == "ё" and b == "е") or (a == "е" and b == "ё"):
                common_prefix += 1
            else:
                break
        if len(original) > 3 and common_prefix < len(original) * 0.5:
            return False
        return True

    # ══════════════════════════════════════════════════════════════
    # Natasha-анализ
    # ══════════════════════════════════════════════════════════════

    def _analyze_phrase(self, phrase: str) -> Optional[PhraseAnalysis]:
        if self._ensure_natasha():
            return self._analyze_with_natasha(phrase)
        return self._analyze_heuristic(phrase)

    def _analyze_with_natasha(self, phrase: str) -> Optional[PhraseAnalysis]:
        synthetic = f"Это {phrase}."
        doc = Doc(synthetic)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        doc.parse_syntax(self._syntax_parser)
        for t in doc.tokens:
            t.lemmatize(self._morph_vocab)

        start_offset = synthetic.index(phrase)
        end_offset = start_offset + len(phrase)

        tokens = []
        word_idx = 0
        for nt in doc.tokens:
            if nt.start < start_offset or nt.stop > end_offset:
                continue
            feats = dict(nt.feats) if isinstance(nt.feats, dict) else {}
            tid = nt.id
            token = TokenInfo(
                idx=word_idx,
                id=hash(tid) % 100000, text=nt.text,
                lemma=nt.lemma or nt.text.lower(),
                pos=nt.pos or "", dep_rel=nt.rel or "",
                head_id=hash(nt.head_id) % 100000 if nt.head_id else 0,
                feats=feats)
            tokens.append(token)
            word_idx += 1

        if not tokens:
            return None

        head_idx = self._find_head(tokens)
        head = tokens[head_idx]

        # Собираем nmod-поддеревья
        nmod_subtree_ids = set()
        self._collect_nmod_subtree(tokens, head, nmod_subtree_ids)

        # Проверяем: голова — топоним-классификатор? (P1)
        head_is_classifier = head.lemma.lower() in _TOPONYM_CLASSIFIERS

        for i, t in enumerate(tokens):
            if i == head_idx:
                t.should_inflect = True
                t.dep_rel = "root"
            elif t.id in nmod_subtree_ids:
                t.should_inflect = False
            elif t.dep_rel in ("amod", "det"):
                t.should_inflect = True
            elif t.dep_rel in ("flat:name", "flat"):
                t.should_inflect = True
            elif t.dep_rel == "appos":
                if self._is_quoted(t.text):
                    t.should_inflect = False
                elif head_is_classifier and self._is_indeclinable_toponym(t.text):
                    t.should_inflect = False  # P1: «Республика Татарстан»
                else:
                    t.should_inflect = True
            elif t.dep_rel == "nummod":
                t.should_inflect = True
            elif t.dep_rel == "case" or t.pos in ("ADP", "CCONJ", "SCONJ"):
                t.should_inflect = False
            elif t.dep_rel in ("nmod", "obl", "nsubj", "obj"):
                t.should_inflect = False
            elif t.pos == "PUNCT":
                t.should_inflect = False
            else:
                # Для неизвестного dep_rel: если NOUN после головы → nmod, иначе не склоняем
                t.should_inflect = False

        hg, hn, ha = self._get_morph_from_feats_or_pymorphy(head)
        return PhraseAnalysis(tokens=tokens, head_idx=head_idx,
                              head_gender=hg, head_number=hn, head_animacy=ha)

    def _collect_nmod_subtree(self, tokens, head, result):
        for t in tokens:
            if t.head_id == head.id and t.dep_rel in ("nmod", "obl", "case"):
                result.add(t.id)
                self._collect_subtree(tokens, t, result)

    def _collect_subtree(self, tokens, root, result):
        for t in tokens:
            if t.head_id == root.id and t.id not in result:
                result.add(t.id)
                self._collect_subtree(tokens, t, result)

    def _find_head(self, tokens: list[TokenInfo]) -> int:
        token_ids = {t.id for t in tokens}
        # 1. Явный root
        for i, t in enumerate(tokens):
            if t.dep_rel == "root":
                return i
        # 2. Токен, чей head вне фразы
        candidates = []
        for i, t in enumerate(tokens):
            if t.head_id not in token_ids:
                candidates.append(i)
        # Из кандидатов предпочитаем NOUN
        for i in candidates:
            if tokens[i].pos == "NOUN":
                return i
        if candidates:
            return candidates[0]
        # 3. Любой NOUN
        for i, t in enumerate(tokens):
            if t.pos == "NOUN":
                return i
        return 0

    # ══════════════════════════════════════════════════════════════
    # Эвристический анализ — v3 с NUMR/ANUM + топонимами
    # ══════════════════════════════════════════════════════════════

    def _analyze_heuristic(self, phrase: str) -> Optional[PhraseAnalysis]:
        morph = self.morph_engine.morph
        words = phrase.split()
        tokens = []
        head_idx = -1
        in_pp = False

        for i, word in enumerate(words):
            clean = re.sub(r'[«»""\'().,;:!?]', '', word)
            if not clean:
                tokens.append(TokenInfo(idx=i, id=i+1, text=word, pos="PUNCT",
                                        dep_rel="punct", should_inflect=False))
                continue

            if clean.lower() in _PREPOSITIONS:
                in_pp = True
                tokens.append(TokenInfo(idx=i, id=i+1, text=word, lemma=clean.lower(),
                                        pos="ADP", dep_rel="case", should_inflect=False))
                continue

            # Нормализуем ё для pymorphy-парсинга
            parse_word = _normalize_yo(clean)
            parses = morph.parse(parse_word)
            if not parses:
                parses = morph.parse(clean)  # fallback: попробуем оригинал
            if not parses:
                tokens.append(TokenInfo(idx=i, id=i+1, text=word, pos="X",
                                        dep_rel="dep", should_inflect=False))
                continue

            best = parses[0]
            pos = str(best.tag.POS) if best.tag.POS else "X"
            lemma = best.normal_form

            token = TokenInfo(idx=i, id=i+1, text=word, lemma=lemma, pos=pos)
            tokens.append(token)

            if in_pp:
                token.dep_rel = "nmod"
                token.should_inflect = False
                continue

            # Ищем первое существительное как голову
            if pos == "NOUN" and head_idx < 0:
                head_idx = i
                token.dep_rel = "root"
                token.should_inflect = True
                continue

            # P3: Модификаторы (прилагательные, причастия, порядковые числительные)
            if pos in _MODIFIER_POS and head_idx < 0:
                token.dep_rel = "amod"
                token.should_inflect = True
            elif pos in _MODIFIER_POS and head_idx >= 0 and not in_pp:
                token.dep_rel = "amod"
                token.should_inflect = True
            elif pos == "NOUN" and head_idx >= 0:
                token.dep_rel = "nmod"
                token.should_inflect = False
            else:
                token.should_inflect = False

        if head_idx < 0:
            # Не нашли NOUN: берём первый токен как голову
            head_idx = 0
            if tokens:
                tokens[0].dep_rel = "root"
                tokens[0].should_inflect = True

        # P1: Топоним-классификатор → зависимое не склоняется
        if head_idx < len(tokens):
            head_token = tokens[head_idx]
            if head_token.lemma.lower() in _TOPONYM_CLASSIFIERS:
                for t in tokens:
                    if t.dep_rel == "nmod" and t.text[0].isupper():
                        pass  # уже не склоняется
                    elif t.dep_rel not in ("root", "amod", "det") and t.should_inflect:
                        if self._is_indeclinable_toponym(t.text):
                            t.should_inflect = False
                # Также: appos (NOUN после головы с заглавной буквы)
                for j in range(head_idx + 1, len(tokens)):
                    t = tokens[j]
                    if t.pos == "NOUN" and t.text[0].isupper():
                        if self._is_indeclinable_toponym(t.text):
                            t.dep_rel = "appos"
                            t.should_inflect = False

        hg, hn, ha = (None, None, None)
        if head_idx < len(tokens):
            hg, hn, ha = self._get_morph_from_pymorphy(tokens[head_idx].text)

        return PhraseAnalysis(tokens=tokens, head_idx=head_idx,
                              head_gender=hg, head_number=hn, head_animacy=ha)

    def _heuristic_inflect(self, phrase: str, target_case: Case) -> str:
        analysis = self._analyze_heuristic(phrase)
        if analysis is None:
            return phrase
        for t in analysis.tokens:
            if not t.should_inflect:
                t.inflected = t.text
            elif t.dep_rel == "root":
                t.inflected = self._safe_inflect_word(t.text, target_case)
            elif t.dep_rel == "amod":
                t.inflected = self._inflect_agreement(
                    t.text, target_case,
                    analysis.head_gender, analysis.head_number,
                    analysis.head_animacy)
            else:
                t.inflected = t.text
        return self._reassemble(phrase, analysis.tokens)

    # ══════════════════════════════════════════════════════════════
    # Согласование
    # ══════════════════════════════════════════════════════════════

    def _inflect_agreement(self, word: str, target_case: Case,
                           gender=None, number=None, animacy=None) -> str:
        from ..core.enums import Gender as G, Number as N, Animacy as A
        g = G(gender) if gender else None
        n = N(number) if number else None
        a = A(animacy) if animacy else None

        # Попытка 1: оригинал
        r = self.morph_engine.inflect_with_agreement(word, target_case, g, n, a)
        if r and self._is_sane_inflection(word, r):
            return _restore_yo(word, r)

        # Попытка 2: ё→е
        normalized = _normalize_yo(word)
        if normalized != word:
            r = self.morph_engine.inflect_with_agreement(
                normalized, target_case, g, n, a)
            if r and self._is_sane_inflection(word, r):
                return _restore_yo(word, r)

        # Попытка 3: без согласования, просто склонить
        r2 = self.morph_engine.inflect(word, target_case)
        if r2 and self._is_sane_inflection(word, r2.inflected_form):
            return _restore_yo(word, r2.inflected_form)

        return word

    # ══════════════════════════════════════════════════════════════
    # Морфо-характеристики головы
    # ══════════════════════════════════════════════════════════════

    def _get_morph_from_feats_or_pymorphy(self, head: TokenInfo):
        feats = head.feats
        g = {"Masc": "masc", "Fem": "femn", "Neut": "neut"}.get(feats.get("Gender"))
        n = {"Sing": "sing", "Plur": "plur"}.get(feats.get("Number"))
        a = {"Anim": "anim", "Inan": "inan"}.get(feats.get("Animacy"))
        if g and n:
            return (g, n, a)
        return self._get_morph_from_pymorphy(head.text)

    def _get_morph_from_pymorphy(self, word: str):
        info = self.morph_engine.analyze(word)
        if info is None:
            # Попробуем с ё→е
            info = self.morph_engine.analyze(_normalize_yo(word))
        if info is None:
            return (None, None, None)
        return (info.gender.value if info.gender else None,
                info.number.value if info.number else None,
                info.animacy.value if info.animacy else None)

    # ══════════════════════════════════════════════════════════════
    # P1: Правила для топонимов
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _is_indeclinable_toponym(word: str) -> bool:
        """
        Топоним не склоняется при классификаторе, если:
          - Оканчивается на согласный (тюркские/нерусские: Татарстан, Башкортостан)
          - Оканчивается на -о/-е/-и (Осло, Сочи, Токио)
          - Это аббревиатура (РФ, СССР)
        """
        if not word or not word[0].isupper():
            return False
        low = word.lower()
        last = low[-1]
        # Оканчивается на согласный → вероятно нерусский топоним
        russian_vowels = "аеёиоуыэюя"
        if last not in russian_vowels and last not in "ьъ":
            return True
        # Оканчивается на -о/-е/-и → несклоняемое
        if last in "оеи":
            return True
        # Аббревиатура
        if word.isupper() and len(word) <= 5:
            return True
        return False

    # ══════════════════════════════════════════════════════════════
    # P0: Позиционная сборка (замена str.replace)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _reassemble(original: str, tokens: list[TokenInfo]) -> str:
        """
        Собрать фразу из склонённых токенов по ПОЗИЦИИ, а не str.replace().

        Каждый токен знает свой idx — позицию слова в оригинальной фразе.
        Мы заменяем слово на inflected, сохраняя окружающую пунктуацию.
        """
        orig_words = original.split()

        # Если количество токенов не совпадает с количеством слов,
        # маппим по idx (может быть sparse)
        result_words = list(orig_words)  # копия

        for token in tokens:
            idx = token.idx
            if idx < 0 or idx >= len(result_words):
                continue
            if not token.inflected or token.inflected == token.text:
                continue

            orig_word = orig_words[idx]

            # Выделяем пунктуацию слева и справа из оригинального слова
            # «Газпром» → leading=«, core=Газпром, trailing=»
            leading = ""
            trailing = ""
            i_start = 0
            while i_start < len(orig_word) and not orig_word[i_start].isalnum() and orig_word[i_start] != '-':
                leading += orig_word[i_start]
                i_start += 1
            i_end = len(orig_word) - 1
            while i_end > i_start and not orig_word[i_end].isalnum() and orig_word[i_end] != '-':
                trailing = orig_word[i_end] + trailing
                i_end -= 1

            result_words[idx] = leading + token.inflected + trailing

        return " ".join(result_words)

    @staticmethod
    def _is_quoted(text: str) -> bool:
        return bool(re.search(r'[«»""\']', text))