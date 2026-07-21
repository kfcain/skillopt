"""Deterministic set-based scoring for closed-set classification envs.

Every closed-set env (iacpolicy, grccontrol, cweclass) asks the target model
to emit a set of canonical IDs inside ``<answer>...</answer>``. The functions
here parse that answer into a normalized set and score it against a gold set
with exact-match and F1. Keeping this logic pure and shared means the val gate
compares apples to apples across all three domains.

Design points:
- ``parse_id_set`` upper-cases, strips surrounding punctuation, dedupes, and
  (optionally) restricts to the candidate set shown in the prompt so a
  hallucinated ID cannot inflate recall.
- The empty set is a first-class answer (a clean IaC resource violates nothing;
  safe code has no CWE). ``∅ == ∅`` scores 1.0; ``∅`` vs. non-empty scores 0.0.
- ``EMPTY_SENTINELS`` map explicit "nothing" answers (NONE / SAFE / N/A) to ∅.
"""
from __future__ import annotations

import re

# Tokens a model may use to explicitly mean "the empty set" (no violations,
# safe code, no applicable control). Compared case-insensitively.
EMPTY_SENTINELS: frozenset[str] = frozenset(
    {"NONE", "SAFE", "N/A", "NA", "NULL", "NO", "NOTHING", "NO_VIOLATIONS"}
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Split an answer body on commas, semicolons, or any whitespace.
_SPLIT_RE = re.compile(r"[,\s;]+")
# Punctuation we strip from the edges of a single token (backticks, quotes,
# parentheses, brackets, trailing periods) without touching inner separators
# like the '-' in "AC-7" or '_' in "CKV_AWS_19".
_EDGE_CHARS = "`'\"()[]{}.:*"


def extract_answer(text: str) -> str:
    """Return the content of the last ``<answer>...</answer>`` block.

    Falls back to the last non-empty line, then the whole stripped text. This
    generalizes ``searchqa/evaluator.py:extract_answer``.
    """
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def normalize_id(token: str) -> str:
    """Normalize a single ID token: strip edge punctuation, upper-case."""
    return token.strip().strip(_EDGE_CHARS).strip().upper()


def parse_id_set(
    text: str,
    *,
    allowed: set[str] | list[str] | None = None,
    already_extracted: bool = False,
) -> set[str]:
    """Parse model output into a normalized set of canonical IDs.

    Parameters
    ----------
    text : str
        Raw model response (or an already-extracted answer body if
        ``already_extracted`` is True).
    allowed : iterable of str, optional
        The candidate set shown in the prompt. When given, IDs outside it are
        dropped so hallucinated IDs cannot score. Compared after normalization.
    already_extracted : bool
        Skip ``extract_answer`` when the caller already isolated the answer body.
    """
    body = text if already_extracted else extract_answer(text)
    allowed_norm = {normalize_id(a) for a in allowed} if allowed is not None else None

    ids: set[str] = set()
    for raw in _SPLIT_RE.split(body or ""):
        tok = normalize_id(raw)
        if not tok:
            continue
        if tok in EMPTY_SENTINELS:
            # An explicit "nothing" token contributes no IDs. If it appears
            # alongside real IDs it is simply ignored.
            continue
        if allowed_norm is not None and tok not in allowed_norm:
            continue
        ids.add(tok)
    return ids


def set_em(pred: set[str], gold: set[str]) -> float:
    """Strict exact match over sets. ``∅ == ∅`` is 1.0."""
    return 1.0 if pred == gold else 0.0


def set_f1(pred: set[str], gold: set[str]) -> float:
    """Set-level F1. ``∅`` vs ``∅`` → 1.0; ``∅`` vs non-empty → 0.0."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def build_eval_detail(
    *,
    question: str,
    predicted: set[str],
    gold: set[str],
    em: float,
    f1: float,
    label: str = "canonical IDs",
) -> str:
    """Render the ``[EVALUATION RESULT]`` block appended to conversation.json.

    The reflection analyst reads this to learn *which* IDs were missed or extra,
    which is what turns a wrong answer into a concrete skill edit.
    """
    missed = sorted(gold - predicted)
    extra = sorted(predicted - gold)
    return (
        "[EVALUATION RESULT]\n"
        f"Task: {question}\n"
        f"Predicted {label}: {sorted(predicted)}\n"
        f"Gold {label}: {sorted(gold)}\n"
        f"Missed (in gold, not predicted): {missed}\n"
        f"Extra (predicted, not in gold): {extra}\n"
        f"Set Exact Match: {em}\n"
        f"Set F1: {f1:.4f}"
    )
