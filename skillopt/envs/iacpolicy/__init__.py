"""IaC policy-as-code environment (infrastructure domain).

Tier-1 diagnostic task: given an Infrastructure-as-Code resource block and a
shown candidate set of policy check IDs, predict the set the resource
*violates*. Ground truth is produced by a real policy engine (Checkov/tfsec)
offline and frozen into the split, so the optimizer's gate sees a deterministic
oracle — the cleanest possible scoring signal.
"""
