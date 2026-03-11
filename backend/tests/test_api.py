"""
Tests for the FastAPI endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from saddle.api import app

client = TestClient(app)


class TestSurfaceEndpoint:
    def test_rosenbrock_default(self) -> None:
        resp = client.get("/surface?name=rosenbrock")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rows"] == 100
        assert data["cols"] == 100
        assert len(data["values"]) == 100
        assert len(data["values"][0]) == 100

    def test_custom_resolution(self) -> None:
        resp = client.get("/surface?name=bowl&resolution=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rows"] == 50
        assert data["cols"] == 50

    def test_all_surfaces(self) -> None:
        for name in ("rosenbrock", "beale", "himmelblau", "bowl", "monkey_saddle"):
            resp = client.get(f"/surface?name={name}&resolution=10")
            assert resp.status_code == 200, f"Failed for {name}"

    def test_invalid_resolution(self) -> None:
        resp = client.get("/surface?name=bowl&resolution=1")
        assert resp.status_code == 400


class TestOptimiseEndpoint:
    def test_sgd_bowl(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "bowl",
            "optimiser": "sgd",
            "x0": 3.0, "y0": 4.0,
            "num_steps": 100,
            "lr": 0.01,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["trajectory"]) == 101
        assert data["trajectory"][0]["x"] == 3.0
        assert data["trajectory"][0]["y"] == 4.0
        # Should make progress
        assert data["trajectory"][-1]["loss"] < data["trajectory"][0]["loss"]

    def test_adam_rosenbrock(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "rosenbrock",
            "optimiser": "adam",
            "x0": -1.0, "y0": 1.0,
            "num_steps": 200,
            "lr": 0.01,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"][-1]["loss"] < data["trajectory"][0]["loss"]

    def test_adahessian_bowl(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "bowl",
            "optimiser": "adahessian",
            "x0": 3.0, "y0": 4.0,
            "num_steps": 100,
            "lr": 0.1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"][-1]["loss"] < data["trajectory"][0]["loss"]

    def test_c_adam_bowl(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "bowl",
            "optimiser": "c_adam",
            "x0": 3.0, "y0": 4.0,
            "num_steps": 200,
            "lr": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["trajectory"]) == 201
        assert data["optimiser"] == "c_adam"
        assert data["trajectory"][-1]["loss"] < 0.01

    def test_rmsprop_bowl(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "bowl",
            "optimiser": "rmsprop",
            "x0": 3.0, "y0": 4.0,
            "num_steps": 100,
            "lr": 0.01,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"][-1]["loss"] < data["trajectory"][0]["loss"]

    def test_lbfgs_rosenbrock(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "rosenbrock",
            "optimiser": "lbfgs",
            "x0": -1.0, "y0": 1.0,
            "num_steps": 200,
            "lr": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"][-1]["loss"] < data["trajectory"][0]["loss"]

    def test_trajectory_has_correct_fields(self) -> None:
        resp = client.post("/optimise", json={
            "surface": "bowl",
            "optimiser": "sgd",
            "num_steps": 5,
        })
        data = resp.json()
        point = data["trajectory"][0]
        assert "x" in point
        assert "y" in point
        assert "loss" in point


class TestSurfacesListEndpoint:
    def test_returns_list(self) -> None:
        resp = client.get("/surfaces")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 5

    def test_each_surface_has_fields(self) -> None:
        resp = client.get("/surfaces")
        data = resp.json()
        for surface in data:
            assert "name" in surface
            assert "key" in surface
            assert "formula" in surface
            assert "description" in surface
            assert "minima" in surface


class TestGradientEndpoint:
    def test_gradient_field_shape(self) -> None:
        resp = client.get("/gradient?name=bowl&resolution=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["x"]) == 5
        assert len(data["y"]) == 5
        assert len(data["gx"]) == 5
        assert len(data["gx"][0]) == 5
        assert len(data["gy"]) == 5
        assert len(data["gy"][0]) == 5


class TestBenchmarkEndpoint:
    def test_basic_benchmark(self) -> None:
        resp = client.get("/benchmark?num_steps=500&param_dim=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_steps"] == 500
        assert data["param_dim"] == 2
        assert data["c_total_ms"] > 0
        assert data["jax_total_ms"] > 0
        assert data["speedup"] > 0
