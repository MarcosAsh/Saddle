"""
Learning rate schedule functions.

Each schedule computes the learning rate for a given step, based on
the base learning rate, current step, total steps, and optional warmup.
"""

from __future__ import annotations

import math
from typing import Literal

ScheduleName = Literal["constant", "cosine", "warmup_cosine", "step_decay"]


def compute_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    schedule: ScheduleName = "constant",
    warmup_steps: int = 0,
) -> float:
    """Return the learning rate at *step* under the given schedule."""
    if total_steps <= 0:
        return base_lr

    if schedule == "constant":
        return base_lr

    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    if schedule == "warmup_cosine":
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step / max(warmup_steps, 1))
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    if schedule == "step_decay":
        interval = max(total_steps // 3, 1)
        return base_lr * (0.5 ** (step // interval))

    return base_lr
