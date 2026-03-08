"""
Фразовый движок склонения — v2.

Исправление v1: «общество с ограниченной ответственностью» в дательном давало
  «обществу с ограниченному ответственностью» — ошибочно склонялось прилагательное
  внутри предложной группы.

v2 fix: эвристический парсер распознаёт предлоги (с, в, на, для, и т.д.)
  и помечает всю предложную группу как неизменяемую.
  Также улучшена работа с Natasha: nmod и его поддеревья не склоняются.
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

# Русские предлоги — если встретили, всё после него (до конца или следующего head)
# относится к предложной группе и НЕ склоняется
_PREPOSITIONS = {
    "с", "в", "на", "за", "о", "об", "обо", "по", "к", "ко", "у", "из",
    "от", "до", "для", "при", "про", "без", "над", "под", "между",
    "через", "перед", "после", "около", "среди", "вместо", "ради",
}


@dataclass
class TokenInfo:
    id: int
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

    # ── Главный метод ─────────────────────────────────────────────

    def inflect_phrase(self, phrase: str, target_case: Case) -> str:
        if target_case == Case.NOMINATIVE:
            return phrase
        phrase = phrase.strip()
        if not phrase:
            return phrase

        words = phrase.split()
        if len(words) == 1:
            r = self.morph_engine.inflect(words[0], target_case)
            return r.inflected_form if r else phrase

        analysis = self._analyze_phrase(phrase)
        if analysis is None or analysis.head_idx < 0:
            return self._heuristic_inflect(phrase, target_case)

        head = analysis.tokens[analysis.head_idx]

        for i, token in enumerate(analysis.tokens):
            if not token.should_inflect:
                token.inflected = token.text
                continue

            if token.dep_rel == "root" or token.id == head.id:
                r = self.morph_engine.inflect(token.text, target_case)
                token.inflected = r.inflected_form if r else token.text

            elif token.dep_rel in ("amod", "det"):
                token.inflected = self._inflect_agreement(
                    token.text, target_case,
                    analysis.head_gender, analysis.head_number, analysis.head_animacy)

            elif token.dep_rel in ("flat:name", "flat", "appos", "nummod"):
                r = self.morph_engine.inflect(token.text, target_case)
                token.inflected = r.inflected_form if r else token.text

            else:
                token.inflected = token.text

        return self._reassemble(phrase, analysis.tokens)

    # ── Natasha-анализ ───────────────────────────────────────────

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
        id_map = {}  # natasha_id → our index
        for nt in doc.tokens:
            if nt.start < start_offset or nt.stop > end_offset:
                continue
            feats = dict(nt.feats) if isinstance(nt.feats, dict) else {}
            tid = nt.id
            token = TokenInfo(
                id=hash(tid) % 100000, text=nt.text,
                lemma=nt.lemma or nt.text.lower(),
                pos=nt.pos or "", dep_rel=nt.rel or "",
                head_id=hash(nt.head_id) % 100000 if nt.head_id else 0,
                feats=feats)
            id_map[tid] = len(tokens)
            tokens.append(token)

        if not tokens:
            return None

        head_idx = self._find_head(tokens)
        head = tokens[head_idx]

        # Собираем ID всех токенов, входящих в nmod-поддеревья
        # Все зависимые nmod и их зависимые тоже не склоняются
        nmod_subtree_ids = set()
        self._collect_nmod_subtree(tokens, head, nmod_subtree_ids)

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
                t.should_inflect = not self._is_quoted(t.text)
            elif t.dep_rel == "nummod":
                t.should_inflect = True
            elif t.dep_rel == "case" or t.pos in ("ADP", "CCONJ", "SCONJ"):
                t.should_inflect = False
            elif t.dep_rel in ("nmod", "obl", "nsubj", "obj"):
                t.should_inflect = False
            elif t.pos == "PUNCT":
                t.should_inflect = False
            else:
                t.should_inflect = False

        hg, hn, ha = self._get_morph_from_feats_or_pymorphy(head)
        return PhraseAnalysis(tokens=tokens, head_idx=head_idx,
                              head_gender=hg, head_number=hn, head_animacy=ha)

    def _collect_nmod_subtree(self, tokens: list[TokenInfo], head: TokenInfo,
                               result: set):
        """Рекурсивно собрать id всех токенов в nmod/obl-поддеревьях головы."""
        for t in tokens:
            if t.head_id == head.id and t.dep_rel in ("nmod", "obl", "case"):
                result.add(t.id)
                self._collect_subtree(tokens, t, result)

    def _collect_subtree(self, tokens: list[TokenInfo], root: TokenInfo, result: set):
        for t in tokens:
            if t.head_id == root.id and t.id not in result:
                result.add(t.id)
                self._collect_subtree(tokens, t, result)

    def _find_head(self, tokens: list[TokenInfo]) -> int:
        token_ids = {t.id for t in tokens}
        for i, t in enumerate(tokens):
            if t.dep_rel == "root": return i
            if t.head_id not in token_ids: return i
        for i, t in enumerate(tokens):
            if t.pos == "NOUN": return i
        return 0

    # ══════════════════════════════════════════════════════════════
    # Эвристический анализ — v2 с обработкой предлогов
    # ══════════════════════════════════════════════════════════════

    def _analyze_heuristic(self, phrase: str) -> Optional[PhraseAnalysis]:
        morph = self.morph_engine.morph
        words = phrase.split()
        tokens = []
        head_idx = -1
        in_pp = False  # Флаг: находимся внутри предложной группы

        for i, word in enumerate(words):
            clean = re.sub(r'[«»""\'().,;:!?]', '', word)
            if not clean:
                tokens.append(TokenInfo(id=i+1, text=word, pos="PUNCT",
                                        dep_rel="punct", should_inflect=False))
                continue

            # Проверяем, является ли слово предлогом
            if clean.lower() in _PREPOSITIONS:
                in_pp = True  # Всё после предлога — предложная группа
                tokens.append(TokenInfo(id=i+1, text=word, lemma=clean.lower(),
                                        pos="ADP", dep_rel="case", should_inflect=False))
                continue

            parses = morph.parse(clean)
            if not parses:
                tokens.append(TokenInfo(id=i+1, text=word, pos="X", dep_rel="dep",
                                        should_inflect=False))
                continue

            best = parses[0]
            pos = str(best.tag.POS) if best.tag.POS else "X"

            token = TokenInfo(id=i+1, text=word, lemma=best.normal_form, pos=pos)
            tokens.append(token)

            # Если мы внутри предложной группы — не склоняем
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

            # Прилагательные/причастия до головы — модификаторы
            if pos in ("ADJF", "PRTF", "ADJ") and head_idx < 0:
                token.dep_rel = "amod"
                token.should_inflect = True
            elif pos in ("ADJF", "PRTF", "ADJ") and head_idx >= 0 and not in_pp:
                # Прилагательное после головы, но до предлога — тоже модификатор
                token.dep_rel = "amod"
                token.should_inflect = True
            elif pos == "NOUN" and head_idx >= 0:
                token.dep_rel = "nmod"
                token.should_inflect = False
            else:
                token.should_inflect = False

        if head_idx < 0:
            head_idx = 0
            if tokens:
                tokens[0].dep_rel = "root"
                tokens[0].should_inflect = True

        hg, hn, ha = (None, None, None)
        if head_idx < len(tokens):
            hg, hn, ha = self._get_morph_from_pymorphy(tokens[head_idx].text)

        return PhraseAnalysis(tokens=tokens, head_idx=head_idx,
                              head_gender=hg, head_number=hn, head_animacy=ha)

    def _heuristic_inflect(self, phrase: str, target_case: Case) -> str:
        analysis = self._analyze_heuristic(phrase)
        if analysis is None: return phrase
        for t in analysis.tokens:
            if not t.should_inflect:
                t.inflected = t.text
            elif t.dep_rel == "root":
                r = self.morph_engine.inflect(t.text, target_case)
                t.inflected = r.inflected_form if r else t.text
            elif t.dep_rel == "amod":
                t.inflected = self._inflect_agreement(
                    t.text, target_case,
                    analysis.head_gender, analysis.head_number, analysis.head_animacy)
            else:
                t.inflected = t.text
        return self._reassemble(phrase, analysis.tokens)

    # ── Согласование ─────────────────────────────────────────────

    def _inflect_agreement(self, word: str, target_case: Case,
                            gender=None, number=None, animacy=None) -> str:
        from ..core.enums import Gender as G, Number as N, Animacy as A
        g = G(gender) if gender else None
        n = N(number) if number else None
        a = A(animacy) if animacy else None
        r = self.morph_engine.inflect_with_agreement(word, target_case, g, n, a)
        return r if r else word

    # ── Морфо-характеристики головы ──────────────────────────────

    def _get_morph_from_feats_or_pymorphy(self, head: TokenInfo):
        feats = head.feats
        g = {"Masc":"masc","Fem":"femn","Neut":"neut"}.get(feats.get("Gender"))
        n = {"Sing":"sing","Plur":"plur"}.get(feats.get("Number"))
        a = {"Anim":"anim","Inan":"inan"}.get(feats.get("Animacy"))
        if g and n:
            return (g, n, a)
        return self._get_morph_from_pymorphy(head.text)

    def _get_morph_from_pymorphy(self, word: str):
        info = self.morph_engine.analyze(word)
        if info is None: return (None, None, None)
        return (info.gender.value if info.gender else None,
                info.number.value if info.number else None,
                info.animacy.value if info.animacy else None)

    @staticmethod
    def _is_quoted(text: str) -> bool:
        return bool(re.search(r'[«»""\']', text))

    @staticmethod
    def _reassemble(original: str, tokens: list[TokenInfo]) -> str:
        result = original
        for token in tokens:
            if token.inflected and token.inflected != token.text:
                result = result.replace(token.text, token.inflected, 1)
        return result
