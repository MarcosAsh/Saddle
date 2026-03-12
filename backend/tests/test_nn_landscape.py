"""Tests for neural network loss landscape computation."""

from __future__ import annotations

import pytest

from saddle.nn_landscape import compute_nn_landscape, compute_nn_trajectory


class TestNNLandscape:
    def test_landscape_shape(self) -> None:
        result = compute_nn_landscape(resolution=15, seed=42)
        assert len(result["values"]) == 15
        assert len(result["values"][0]) == 15
        assert result["x_min"] == -1.0
        assert result["x_max"] == 1.0

    def test_landscape_values_finite(self) -> None:
        result = compute_nn_landscape(resolution=10, seed=42)
        for row in result["values"]:
            for v in row:
                assert v >= 0, "Loss should be non-negative"
                assert v < 100, "Loss should be bounded"

    def test_center_is_optimum(self) -> None:
        result = compute_nn_landscape(resolution=11, seed=42)
        center = result["values"][5][5]
        # The center (theta*) should have low loss
        corners = [
            result["values"][0][0],
            result["values"][0][-1],
            result["values"][-1][0],
            result["values"][-1][-1],
        ]
        # Center should generally be lower than at least some corners
        assert center < max(corners)


class TestNNTrajectory:
    def test_trajectory_nonempty(self) -> None:
        traj = compute_nn_trajectory(seed=42)
        assert len(traj) > 10

    def test_trajectory_format(self) -> None:
        traj = compute_nn_trajectory(seed=42)
        for alpha, beta, loss in traj:
            assert isinstance(alpha, float)
            assert isinstance(beta, float)
            assert isinstance(loss, float)
            assert loss >= 0

    def test_loss_decreases(self) -> None:
        traj = compute_nn_trajectory(seed=42)
        # First loss should be higher than last
        assert traj[0][2] > traj[-1][2]
