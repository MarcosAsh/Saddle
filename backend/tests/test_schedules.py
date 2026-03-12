"""Tests for learning rate schedules."""

from __future__ import annotations

import math

import pytest

from saddle.schedules import compute_lr


class TestConstant:
    def test_always_returns_base(self) -> None:
        for step in range(100):
            assert compute_lr(0.01, step, 100, "constant") == 0.01


class TestCosine:
    def test_start_equals_base(self) -> None:
        lr = compute_lr(0.1, 0, 100, "cosine")
        assert lr == pytest.approx(0.1)

    def test_end_near_zero(self) -> None:
        lr = compute_lr(0.1, 100, 100, "cosine")
        assert lr == pytest.approx(0.0, abs=1e-10)

    def test_midpoint(self) -> None:
        lr = compute_lr(0.1, 50, 100, "cosine")
        expected = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 50 / 100))
        assert lr == pytest.approx(expected)

    def test_monotonically_decreasing(self) -> None:
        lrs = [compute_lr(0.1, s, 100, "cosine") for s in range(101)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


class TestWarmupCosine:
    def test_starts_at_zero(self) -> None:
        lr = compute_lr(0.1, 0, 100, "warmup_cosine", warmup_steps=10)
        assert lr == pytest.approx(0.0)

    def test_warmup_linear_ramp(self) -> None:
        lr = compute_lr(0.1, 5, 100, "warmup_cosine", warmup_steps=10)
        assert lr == pytest.approx(0.05)

    def test_peak_at_warmup_end(self) -> None:
        lr = compute_lr(0.1, 10, 100, "warmup_cosine", warmup_steps=10)
        assert lr == pytest.approx(0.1)

    def test_decays_after_warmup(self) -> None:
        lr_peak = compute_lr(0.1, 10, 100, "warmup_cosine", warmup_steps=10)
        lr_later = compute_lr(0.1, 55, 100, "warmup_cosine", warmup_steps=10)
        assert lr_later < lr_peak


class TestStepDecay:
    def test_initial_equals_base(self) -> None:
        lr = compute_lr(0.1, 0, 90, "step_decay")
        assert lr == pytest.approx(0.1)

    def test_first_decay(self) -> None:
        # At step 30 (interval = 90//3 = 30), should be 0.1 * 0.5
        lr = compute_lr(0.1, 30, 90, "step_decay")
        assert lr == pytest.approx(0.05)

    def test_second_decay(self) -> None:
        lr = compute_lr(0.1, 60, 90, "step_decay")
        assert lr == pytest.approx(0.025)
