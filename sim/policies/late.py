"""Late binding (NetKV-style): pick the decode instance after prefill
finishes, using the freshest view, minimising transfer time + decode cost.
The whole KV cache is sent after prefill, so none of it overlaps."""
from __future__ import annotations

from .base import Policy


class LateBinding(Policy):
    name = "late"
    # Uses Policy.on_arrival (returns None) and Policy.on_prefill_done.
