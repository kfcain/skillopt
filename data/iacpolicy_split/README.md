# iacpolicy split

Infrastructure domain, Tier-1 (diagnostic) task: given an Infrastructure-as-Code
resource block and a shown candidate set of policy check IDs, predict the set of
checks the resource **violates**.

## Item schema

```json
{
  "id": "iac_tf_s3_001",
  "task_type": "terraform_aws",
  "iac_snippet": "resource \"aws_s3_bucket\" \"data\" { ... }",
  "candidate_checks": ["CKV_AWS_18", "CKV_AWS_19", "CKV_AWS_20", "CKV_AWS_21"],
  "candidate_check_descriptions": {"CKV_AWS_19": "S3 bucket has server-side encryption at rest configured", "...": "..."},
  "violated_checks": ["CKV_AWS_18", "CKV_AWS_19"]
}
```

- `candidate_checks` is the closed set shown to the model; `violated_checks ⊆
  candidate_checks` is the gold answer.
- The model answers with a comma-separated list of violated check IDs inside
  `<answer>...</answer>`, or `NONE`. Scoring is deterministic set-EM (`hard`) and
  set-F1 (`soft`) — see `skillopt/envs/_shared/setscore.py`.
- Clean resources (empty `violated_checks`) are included on purpose to teach
  precision.

## Provenance

The checked-in split is **hand-authored** illustrative data so the training loop
runs out of the box. Snippets use inline-attribute style so each violation is
judgeable from the block alone, and gold labels are internally consistent with
well-known, stable Checkov AWS check IDs. They are *not* guaranteed to match any
specific Checkov version.

For a faithful, engine-labeled dataset, regenerate with a pinned engine:

```bash
pip install checkov
python scripts/build_iacpolicy_data.py --src path/to/terraform --out data/iacpolicy_split
```

That records the Checkov version in `split_manifest.json` and freezes the
engine's pass/fail verdicts as gold. Only check IDs and resource snippets are
stored; supply your own check descriptions via `--descriptions` rather than
redistributing vendor rule text.
