"""IaC policy rollout — single-turn misconfiguration detection.

The target model is shown one IaC resource block plus a candidate list of policy
check IDs (each with a short description) and must return the set of check IDs
the resource violates, inside ``<answer>...</answer>``. Scoring is deterministic
set-EM / set-F1 against the frozen engine labels.
"""
from __future__ import annotations

from skillopt.envs._shared import build_eval_detail, parse_id_set, set_em, set_f1
from skillopt.envs._shared.closedset import run_batch as _run_closed_set_batch
from skillopt.prompts import load_prompt

ENV_NAME = "iacpolicy"
DEFAULT_TASK_TYPE = "iac_policy"


def _render_candidates(item: dict) -> str:
    checks = item.get("candidate_checks") or []
    descriptions = item.get("candidate_check_descriptions") or {}
    lines = []
    for check in checks:
        desc = descriptions.get(check, "")
        lines.append(f"- {check}: {desc}" if desc else f"- {check}")
    return "\n".join(lines)


def build_prompts(item: dict, skill_content: str) -> tuple[str, str]:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    system = load_prompt("rollout_system", env=ENV_NAME).format(skill_section=skill_section)

    resource_kind = item.get("task_type") or DEFAULT_TASK_TYPE
    user = (
        f"## Resource ({resource_kind})\n"
        f"```\n{item.get('iac_snippet', '').strip()}\n```\n\n"
        f"## Candidate policy checks\n"
        f"{_render_candidates(item)}\n\n"
        "## Task\n"
        "Return the check IDs from the candidate list that this resource "
        "VIOLATES. If it violates none, answer NONE."
    )
    return system, user


def score(response: str, item: dict) -> dict:
    candidates = item.get("candidate_checks") or []
    gold = parse_id_set(
        ",".join(item.get("violated_checks") or []),
        allowed=candidates,
        already_extracted=True,
    )
    pred = parse_id_set(response, allowed=candidates)
    em = set_em(pred, gold)
    f1 = set_f1(pred, gold)
    detail = build_eval_detail(
        question=f"IaC {item.get('task_type', DEFAULT_TASK_TYPE)} resource",
        predicted=pred,
        gold=gold,
        em=em,
        f1=f1,
        label="check IDs",
    )
    result = {
        "hard": int(em),
        "soft": f1,
        "predicted_answer": ", ".join(sorted(pred)) or "NONE",
        "gold_answer": sorted(gold),
        "eval_detail": detail,
    }
    if em < 1.0:
        result["fail_reason"] = (
            f"set-EM=0: predicted {sorted(pred)} but gold is {sorted(gold)}"
        )
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    workers: int = 8,
    exec_timeout: int = 120,
    max_completion_tokens: int = 4096,
    task_timeout: int = 600,
) -> list[dict]:
    return _run_closed_set_batch(
        items=items,
        out_root=out_root,
        skill_content=skill_content,
        build_prompts=build_prompts,
        score=score,
        default_task_type=DEFAULT_TASK_TYPE,
        env_name=ENV_NAME,
        workers=workers,
        exec_timeout=exec_timeout,
        max_completion_tokens=max_completion_tokens,
        task_timeout=task_timeout,
    )
