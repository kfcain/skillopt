"""Single-turn rollout harness for closed-set classification envs.

This factors the parts that are identical across iacpolicy / grccontrol /
cweclass — the ThreadPool, resume-from-``results.jsonl``, and the persisted
``predictions/<id>/conversation.json`` that the shared reflection stage reads —
so each env's ``rollout.py`` only has to supply two small callables:

- ``build_prompts(item, skill_content) -> (system, user)``
- ``score(response_text, item) -> dict``  with at least ``hard`` and ``soft``;
  optionally ``predicted_answer``, ``fail_reason``, and ``eval_detail`` (the
  ``[EVALUATION RESULT]`` block appended to the conversation for the analyst).

This is the Tier-1 (diagnostic) harness: a single ``chat_target`` call per item.
The agentic Tier-2 (remediation / self-heal) harness lives separately because it
needs an exec backend and an environment to mutate.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

BuildPrompts = Callable[[dict, str], tuple[str, str]]
ScoreFn = Callable[[str, dict], dict]


def _raise_on_systemic_failure(results: list[dict], env_name: str) -> None:
    """Abort when every row failed before the agent produced a response.

    This surfaces a misconfigured target backend as a clear error instead of a
    silently all-zero rollout that would look like a data problem.
    """
    if not results or not all(row.get("agent_ok") is False for row in results):
        return
    reasons = Counter(str(row.get("fail_reason") or "unknown error") for row in results)
    common_reason, count = reasons.most_common(1)[0]
    raise RuntimeError(
        f"{env_name} rollout failed for all {len(results)} items before an agent "
        f"response ({count}x): {common_reason}"
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    build_prompts: BuildPrompts,
    score: ScoreFn,
    default_task_type: str,
    exec_timeout: int = 120,
    max_completion_tokens: int = 4096,
) -> dict:
    """Run one closed-set item: single chat turn, score, persist trajectory."""
    # Imported lazily so the env registers (and scoring stays unit-testable)
    # even where the chat backend SDK is not installed; it is only needed here,
    # at actual rollout time.
    from skillopt.model import chat_target

    item_id = str(item["id"])
    task_type = str(item.get("task_type") or default_task_type)
    result = {
        "id": item_id,
        "task_type": task_type,
        "task_description": item.get("task_description") or item.get("question") or "",
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        system, user = build_prompts(item, skill_content)
        response, _usage = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = 1

        scored = score(response, item)
        result.update(scored)
        result["hard"] = scored.get("hard", 0)
        result["soft"] = scored.get("soft", 0.0)

        conversation = [
            {"type": "message", "role": "system", "content": system},
            {"type": "message", "role": "user", "content": user},
            {"type": "message", "role": "assistant", "content": response},
        ]
        eval_detail = scored.get("eval_detail")
        if eval_detail:
            conversation.append({"role": "system", "content": eval_detail})

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"

    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    build_prompts: BuildPrompts,
    score: ScoreFn,
    default_task_type: str,
    env_name: str,
    workers: int = 8,
    exec_timeout: int = 120,
    max_completion_tokens: int = 4096,
    task_timeout: int = 600,
) -> list[dict]:
    """Run a batch of closed-set items with a ThreadPool. Resume-aware."""
    task_timeout = max(int(task_timeout), int(exec_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        _raise_on_systemic_failure(existing, env_name)
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)
    started_at: dict[str, float] = {}

    def _timeout_result(item: dict, phase: str, reason: str) -> dict:
        return {
            "id": str(item["id"]),
            "task_type": str(item.get("task_type") or default_task_type),
            "task_description": item.get("task_description") or item.get("question") or "",
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "response": "",
            "fail_reason": reason,
            "agent_ok": False,
            "n_turns": 0,
            "phase": phase,
        }

    def _run_one(item: dict) -> dict:
        started_at[str(item["id"])] = time.time()
        return process_one(
            item,
            out_root,
            skill_content,
            build_prompts=build_prompts,
            score=score,
            default_task_type=default_task_type,
            exec_timeout=exec_timeout,
            max_completion_tokens=max_completion_tokens,
        )

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _timeout_result(
                            item, "error", f"unexpected: {type(exc).__name__}: {exc}"
                        )
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct_count += 1
                    acc = correct_count / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} (acc={acc:.3f}) "
                        f"id={res['id']} hard={res.get('hard', '?')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    # cancel() only stops futures that have not started; a running
                    # task keeps going in the background until it returns. Note it
                    # so background consumption is diagnosable.
                    if not fut.cancel():
                        print(
                            f"    [rollout] timed-out task id={futs[fut]['id']} "
                            "already running; it will finish in the background",
                            flush=True,
                        )
                    res = _timeout_result(futs[fut], "timeout", f"task-timeout-{task_timeout}s")
                    results.append(res)
                    completed += 1
                    acc = correct_count / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} (acc={acc:.3f}) "
                        f"id={res['id']} TIMEOUT",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    _raise_on_systemic_failure(results, env_name)
    return results
