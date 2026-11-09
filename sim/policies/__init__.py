"""Binding policies (plan §3.10).

Every policy implements the same interface:

* ``on_arrival(req, view)`` -> decode instance to bind early, or ``None``
* ``on_prefill_done(req, view)`` -> decode instance (only called if unbound)

Prefill placement is shared (least estimated wait), so policies differ only
in *when* and *how* they bind the decode side.
"""
from __future__ import annotations

from .base import Policy
from .early import EarlyBinding
from .late import LateBinding
from .threshold import ThresholdBinding

POLICIES = {
    "early": EarlyBinding,
    "late": LateBinding,
    "threshold": ThresholdBinding,
}


def make_policy(name: str, sim, cfg: dict) -> Policy:
    try:
        return POLICIES[name](sim, cfg)
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
