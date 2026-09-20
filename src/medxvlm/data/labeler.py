"""Rule-based CheXpert-style labeler for free-text chest X-ray reports.

Mirrors the structure of the original CheXpert labeler (mention detection,
negation, uncertainty, aggregation) at a scale appropriate for a single
free-text field rather than a full NegBio pipeline: regex phrase matching,
bounded token-window negation/hedging, and per-report priority aggregation
across mentions.

Label values follow the CheXpert convention:
    1.0  -- positive
    0.0  -- negative
   -1.0  -- uncertain
    NaN  -- not mentioned

See `scripts/validate_labeler.py` for the hand-written assertion suite this
module is validated against before being run over the full corpus.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from medxvlm.models.labels import CHEXPERT_LABELS

POSITIVE = 1.0
NEGATIVE = 0.0
UNCERTAIN = -1.0
NOT_MENTIONED = math.nan

# Bounded token windows: how far a cue may sit from a mention and still
# qualify it. Post-modifier windows are short on purpose -- a post-modifier
# only qualifies what immediately precedes it. Pre-cue windows are longer
# because a single leading cue governs a whole coordinated list ("no focal
# consolidation, pleural effusion, or pneumothorax"); a cue's scope is cut
# off early by SCOPE_RESET_TOKENS (see _pre_cue_in_scope).
NEGATION_PRE_WINDOW = 12
NEGATION_POST_WINDOW = 4
UNCERTAIN_PRE_WINDOW = 8
UNCERTAIN_POST_WINDOW = 3
HISTORICAL_PRE_WINDOW = 6

# Tokens that end a pre-cue's scope when they sit between the cue and a
# later mention: "no pneumothorax and there is a small effusion" must not
# let "no" reach "effusion", while "no consolidation, effusion, or
# pneumothorax" (no reset token in between) must.
SCOPE_RESET_TOKENS = frozenset({
    "is", "are", "was", "were", "there", "has", "have",
    "shows", "show", "showing", "demonstrates", "demonstrate", "demonstrating",
    "noted", "seen", "present", "identified", "which", "that",
    "suggests", "suggesting", "represents", "representing",
})

PRE_NEGATION_CUES = (
    "no", "not", "without", "free of", "absence of", "no evidence of",
    "no signs of", "no sign of", "negative for", "rule out", "ruled out",
    "no findings of", "clear of",
)
POST_NEGATION_CUES = (
    "is absent", "are absent", "has resolved", "have resolved",
    "is ruled out", "was ruled out", "is not present", "are not present",
    "is negative", "has cleared", "have cleared",
)
PRE_UNCERTAIN_CUES = (
    "possible", "probable", "likely", "may represent", "may reflect",
    "suspicious for", "concerning for", "cannot exclude", "can not exclude",
    "questionable", "could represent", "may be",
)
POST_UNCERTAIN_CUES = (
    "cannot be excluded", "can not be excluded", "is suspected",
    "is possible", "is questionable", "vs.", "versus", "or artifact",
)
HISTORICAL_CUES = (
    "history of", "h/o", "previous", "previously", "prior",
)
# Scope-breaking conjunctions: a mention's negation/hedge/historical windows
# never cross these -- each clause is scored independently.
SCOPE_BREAKERS = ("but", "however", "although", "except", "aside from", "apart from")

# Per-label mention phrases. Deliberately compact (not the full NegBio
# vocabulary) but covers the phrasing actually seen in the Indiana CXR
# reports (see docs/project_proposal.md for the dataset).
LABEL_PHRASES: dict[str, tuple[str, ...]] = {
    "No Finding": (),  # derived, not phrase-matched -- see aggregate_report
    "Enlarged Cardiomediastinum": (
        "enlarged mediastinum", "mediastinal widening", "widened mediastinum",
        "enlarged cardiomediastinal silhouette",
    ),
    "Cardiomegaly": (
        "cardiomegaly", "enlarged heart", "enlarged cardiac silhouette",
        "cardiac enlargement",
    ),
    "Lung Opacity": (
        "opacity", "opacities", "opacification", "infiltrate", "infiltrates",
    ),
    "Lung Lesion": (
        "nodule", "nodules", "mass", "lesion", "cavitary lesion",
    ),
    "Edema": (
        "edema", "pulmonary edema", "vascular congestion",
    ),
    "Consolidation": (
        "consolidation", "consolidative opacity",
    ),
    "Pneumonia": (
        "pneumonia", "pneumonic infiltrate",
    ),
    "Atelectasis": (
        "atelectasis", "atelectatic",
    ),
    "Pneumothorax": (
        "pneumothorax",
    ),
    "Pleural Effusion": (
        "pleural effusion", "pleural effusions", "effusion", "effusions",
    ),
    "Pleural Other": (
        "pleural thickening", "pleural scarring", "pleural plaque",
    ),
    "Fracture": (
        "fracture", "fractures",
    ),
    "Support Devices": (
        "catheter", "pacemaker", "sternotomy wire", "sternotomy wires",
        "endotracheal tube", "chest tube", "picc line", "support device",
        "hardware",
    ),
}


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Clause:
    tokens: tuple[_Token, ...]
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Mention:
    """One phrase occurrence of a label, scored within its own clause."""

    label: str
    phrase: str
    value: float
    sentence: str


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-/.]*")
_SENTENCE_SPLIT_RE = re.compile(r"[.;\n]+")


def _tokenize(text: str) -> tuple[_Token, ...]:
    return tuple(_Token(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


_LEADING_ENUM_RE = re.compile(r"^\s*(\d+[.)]\s*)+")


def split_report_sentences(text: str, min_words: int = 3) -> list[str]:
    """Report -> list of clean, de-duplicated sentences suitable as contrastive text units.

    # why: the sentence-level Stage 1 (RadZero-style) aligns an image with
    # each *sentence* of its report rather than the whole report, so the
    # training text distribution matches the short prompts used for
    # zero-shot classification ("There is cardiomegaly."). Enumeration
    # prefixes ("1. ") and fragments under `min_words` carry no findings.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(text):
        sentence = _LEADING_ENUM_RE.sub("", sentence).strip()
        if len(sentence.split()) < min_words:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def _split_clauses(sentence: str) -> list[_Clause]:
    """Split one sentence into clauses at scope-breaking conjunctions.

    # why: negation/hedge/historical windows must not cross a scope-breaking
    # conjunction ("no pneumothorax, but there is a pleural effusion" must
    # not let "no" negate "effusion") -- scoring each clause independently
    # is a simple, testable way to enforce that.
    """
    tokens = _tokenize(sentence)
    breaker_positions = {t.start for t in tokens if t.text in SCOPE_BREAKERS}
    if not breaker_positions:
        return [_Clause(tokens, 0, len(sentence))]

    clauses: list[_Clause] = []
    current: list[_Token] = []
    clause_start = 0
    for token in tokens:
        if token.start in breaker_positions:
            if current:
                clauses.append(_Clause(tuple(current), clause_start, token.start))
            current = []
            clause_start = token.end
            continue
        current.append(token)
    if current:
        clauses.append(_Clause(tuple(current), clause_start, len(sentence)))
    return clauses


def _phrase_present(tokens: tuple[_Token, ...], cues: tuple[str, ...]) -> bool:
    """Whether any cue phrase (single- or multi-word) appears in `tokens`, on word boundaries.

    # why word boundaries, not substring: "no" must not match inside
    # "nodular" or "normally" -- a plain substring test negated every
    # "nodular opacity" and hedged nothing correctly.
    """
    if not tokens:
        return False
    window_text = " ".join(t.text for t in tokens)
    return any(re.search(r"\b" + re.escape(cue) + r"\b", window_text) for cue in cues)


def _pre_cue_in_scope(pre_tokens: tuple[_Token, ...], cues: tuple[str, ...]) -> bool:
    """Whether a pre-cue in `pre_tokens` still governs the token right after them.

    Finds the *last* cue occurrence in the window and requires that no
    SCOPE_RESET_TOKENS sit between the end of that cue and the mention.
    """
    if not pre_tokens:
        return False
    words = [t.text for t in pre_tokens]
    last_cue_end = -1
    for cue in cues:
        cue_words = cue.split()
        n = len(cue_words)
        for i in range(len(words) - n, -1, -1):
            if words[i : i + n] == cue_words:
                last_cue_end = max(last_cue_end, i + n)
                break
    if last_cue_end < 0:
        return False
    return not any(w in SCOPE_RESET_TOKENS for w in words[last_cue_end:])


def _find_phrase_spans(clause_text: str, phrase: str) -> list[tuple[int, int]]:
    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    return [(m.start(), m.end()) for m in re.finditer(pattern, clause_text, flags=re.IGNORECASE)]


def _score_mention(clause: _Clause, clause_text: str, mention_start: int, mention_end: int) -> float:
    tokens = clause.tokens
    # Start index (first token of the phrase) anchors the *pre* windows;
    # end index (first token *after* the phrase) anchors the *post* windows.
    # For a multi-word phrase these differ -- anchoring both to the start
    # index would wrongly fold the phrase's own trailing word(s) into the
    # post-window text.
    start_idx = next((i for i, t in enumerate(tokens) if t.start >= mention_start), len(tokens))
    end_idx = sum(1 for t in tokens if t.start < mention_end)

    pre_tokens = tokens[max(0, start_idx - NEGATION_PRE_WINDOW) : start_idx]
    pre_tokens_hedge = tokens[max(0, start_idx - UNCERTAIN_PRE_WINDOW) : start_idx]
    pre_tokens_hist = tokens[max(0, start_idx - HISTORICAL_PRE_WINDOW) : start_idx]

    # Post windows are anchored to the mention's own end offset -- never the
    # sentence end -- so a hedge later in the clause cannot retroactively
    # qualify an earlier, unrelated mention.
    post_tokens_neg = tokens[end_idx : end_idx + NEGATION_POST_WINDOW]
    post_tokens_hedge = tokens[end_idx : end_idx + UNCERTAIN_POST_WINDOW]

    if _pre_cue_in_scope(pre_tokens, PRE_NEGATION_CUES) or _phrase_present(post_tokens_neg, POST_NEGATION_CUES):
        return NEGATIVE
    if _pre_cue_in_scope(pre_tokens_hedge, PRE_UNCERTAIN_CUES) or _phrase_present(post_tokens_hedge, POST_UNCERTAIN_CUES):
        return UNCERTAIN
    if _phrase_present(pre_tokens_hist, HISTORICAL_CUES):
        return UNCERTAIN
    return POSITIVE


def find_mentions(text: str) -> list[Mention]:
    """Find every label mention in `text`, each scored independently."""
    mentions: list[Mention] = []
    for sentence in _split_sentences(text):
        for clause in _split_clauses(sentence):
            clause_text = sentence[clause.char_start : clause.char_end]
            for label, phrases in LABEL_PHRASES.items():
                for phrase in phrases:
                    for start, end in _find_phrase_spans(clause_text, phrase):
                        value = _score_mention(clause, clause_text, start, end)
                        mentions.append(Mention(label=label, phrase=phrase, value=value, sentence=sentence))
    return mentions


# Priority when a label has multiple, possibly conflicting, mentions across
# a report: a positive mention anywhere wins, matching the original
# CheXpert labeler's aggregation convention.
_VALUE_PRIORITY = {POSITIVE: 3, UNCERTAIN: 2, NEGATIVE: 1}


def aggregate_report(text: str) -> dict[str, float]:
    """Label one report's full text against all 14 CheXpert categories."""
    mentions = find_mentions(text)
    labels: dict[str, float] = {name: NOT_MENTIONED for name in CHEXPERT_LABELS if name != "No Finding"}

    for mention in mentions:
        current = labels.get(mention.label, NOT_MENTIONED)
        if math.isnan(current) or _VALUE_PRIORITY[mention.value] > _VALUE_PRIORITY[current]:
            labels[mention.label] = mention.value

    no_active_finding = all(
        math.isnan(v) or v in (NEGATIVE,) for v in labels.values()
    )
    labels["No Finding"] = POSITIVE if no_active_finding else NEGATIVE
    return {name: labels[name] for name in CHEXPERT_LABELS}
