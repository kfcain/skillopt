"""Shared utilities for closed-set classification environments.

The GRC / InfoSec / infrastructure envs all share the same shape: the target
model is shown a small candidate set and must answer with a set of canonical
IDs inside ``<answer>...</answer>`` tags. Scoring is pure set logic so the
optimizer's accept/reject gate stays deterministic ("noisy scoring kills the
optimizer"). This package centralizes that logic so it is written and reviewed
exactly once.
"""
from __future__ import annotations

from skillopt.envs._shared.setscore import (
    build_eval_detail,
    extract_answer,
    parse_id_set,
    set_em,
    set_f1,
)

__all__ = [
    "build_eval_detail",
    "extract_answer",
    "parse_id_set",
    "set_em",
    "set_f1",
]
