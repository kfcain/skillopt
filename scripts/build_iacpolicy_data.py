#!/usr/bin/env python3
"""Materialize the iacpolicy dataset from real policy-engine output.

The env ships a small *hand-authored* starter split so the training loop is
runnable out of the box. This script produces the faithful version: it runs
Checkov over a directory of Terraform files and freezes the engine's pass/fail
verdicts into ``data/iacpolicy_split/{train,val,test}/items.json``.

Because the labels come from the engine (not a human), the optimizer's val gate
sees a deterministic oracle. Pin the Checkov version when you build so labels are
reproducible; the pinned version is recorded in the split manifest.

Usage
-----
    pip install checkov
    python scripts/build_iacpolicy_data.py --src path/to/terraform \\
        --out data/iacpolicy_split --max-candidates 6 --ratio 6:2:2

``--src`` may be any tree of ``.tf`` files — e.g. the Apache-2.0 example
fixtures under ``tests/`` in the ``bridgecrewio/checkov`` repo. Only the check
IDs and your own resource snippets are stored; vendor rule descriptions are not
redistributed (supply your own via ``--descriptions`` if desired).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict


def run_checkov(src: str) -> dict:
    """Run Checkov over *src* and return parsed JSON results."""
    try:
        proc = subprocess.run(
            ["checkov", "-d", src, "-o", "json", "--compact", "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("checkov not found on PATH. Install with: pip install checkov")
    out = proc.stdout.strip()
    if not out:
        sys.exit(f"checkov produced no JSON output.\nstderr:\n{proc.stderr}")
    data = json.loads(out)
    # checkov may emit a list of framework result blocks or a single block.
    return data[0] if isinstance(data, list) else data


def checkov_version() -> str:
    try:
        return subprocess.run(
            ["checkov", "--version"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _snippet_for(file_path: str, line_range: list[int] | None, src_root: str) -> str:
    """Best-effort extraction of the resource block text from its file."""
    abs_path = file_path if os.path.isabs(file_path) else os.path.join(src_root, file_path.lstrip("/"))
    try:
        lines = open(abs_path, encoding="utf-8").read().splitlines()
    except OSError:
        return ""
    if not line_range or len(line_range) != 2:
        return "\n".join(lines)
    start, end = line_range
    return "\n".join(lines[max(0, start - 1): end])


def build_items(result: dict, src_root: str, descriptions: dict, max_candidates: int) -> list[dict]:
    """Group Checkov check records per resource into closed-set items."""
    checks = result.get("results", {})
    per_resource: dict[str, dict] = defaultdict(
        lambda: {"passed": set(), "failed": set(), "meta": None}
    )
    for status, key in (("passed", "passed_checks"), ("failed", "failed_checks")):
        for rec in checks.get(key, []) or []:
            res = rec.get("resource") or f"{rec.get('file_path')}::{rec.get('resource')}"
            per_resource[res][status].add(rec.get("check_id"))
            if per_resource[res]["meta"] is None:
                per_resource[res]["meta"] = rec

    items: list[dict] = []
    rng = random.Random(42)
    for res, info in sorted(per_resource.items()):
        failed = sorted(c for c in info["failed"] if c)
        passed = sorted(c for c in info["passed"] if c)
        candidates = list(failed)
        # Add passed checks as distractors, capped, so it's a real closed set.
        distractors = [c for c in passed if c not in candidates]
        rng.shuffle(distractors)
        room = max(0, max_candidates - len(candidates))
        candidates += distractors[:room]
        candidates = sorted(set(candidates))
        if not candidates:
            continue
        meta = info["meta"] or {}
        snippet = _snippet_for(
            meta.get("file_path", ""), meta.get("file_line_range"), src_root
        )
        items.append(
            {
                "id": res.replace(".", "_").replace("/", "_"),
                "task_type": "terraform_aws",
                "iac_snippet": snippet,
                "candidate_checks": candidates,
                "candidate_check_descriptions": {
                    c: descriptions.get(c, "") for c in candidates
                },
                "violated_checks": [c for c in failed if c in candidates],
            }
        )
    return items


def split_and_write(items: list[dict], out: str, ratio: str, checkov_ver: str) -> None:
    a, b, c = (int(x) for x in ratio.split(":"))
    rng = random.Random(42)
    rng.shuffle(items)
    n = len(items)
    n_tr = n * a // (a + b + c)
    n_va = n * b // (a + b + c)
    splits = {
        "train": items[:n_tr],
        "val": items[n_tr: n_tr + n_va],
        "test": items[n_tr + n_va:],
    }
    for name, rows in splits.items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "items.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "iacpolicy",
                "oracle": "checkov",
                "checkov_version": checkov_ver,
                "counts": {k: len(v) for k, v in splits.items()},
                "note": "Labels are frozen Checkov verdicts; re-pin the version to reproduce.",
            },
            f,
            indent=2,
        )
    print(f"wrote {n} items -> {out}  (checkov {checkov_ver})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Directory tree of .tf files")
    ap.add_argument("--out", default="data/iacpolicy_split")
    ap.add_argument("--ratio", default="6:2:2", help="train:val:test")
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--descriptions", default="", help="Optional JSON map check_id -> your own description")
    args = ap.parse_args()

    descriptions = {}
    if args.descriptions:
        descriptions = json.load(open(args.descriptions, encoding="utf-8"))

    result = run_checkov(args.src)
    ver = checkov_version()
    items = build_items(result, args.src, descriptions, args.max_candidates)
    if not items:
        sys.exit("No items built — check that --src contains scannable .tf files.")
    split_and_write(items, args.out, args.ratio, ver)


if __name__ == "__main__":
    main()
