"""IaC policy dataloader.

Each split directory (train/, val/, test/) holds a single ``items.json`` — a
JSON array of IaC-check items. The default
:meth:`skillopt.datasets.base.SplitDataLoader.load_split_items` already reads
that array, and the base ``load_raw_items`` handles ``split_mode=ratio``. So
this subclass needs no overrides; it exists to give the env a named loader
class (mirroring the other envs).
"""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class IacPolicyDataLoader(SplitDataLoader):
    """Dataloader for the IaC policy-as-code benchmark."""
