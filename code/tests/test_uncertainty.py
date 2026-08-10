"""Tests for the conformal uncertainty module (Formula E)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mvgt_net.uncertainty import ConformalPredictor, ConformalConfig


class DummyModel(torch.nn.Module):
    """Simple model: returns input * 2 + 1 (deterministic)."""
    def forward(self, x):
        return x * 2.0 + 1.0


def test_conformal_config_validation():
    """Config validation should reject invalid alpha and method."""
    with pytest.raises(ValueError):
        ConformalConfig(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalConfig(alpha=1.0)
    with pytest.raises(ValueError):
        ConformalConfig(alpha=0.5, method="invalid")


def test_conformal_predictor_calibration():
    """Calibration should compute q_hat from the calibration set."""
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=10))
    # Generate 100 calibration samples
    x = torch.randn(100, 5)
    y_true = x * 2.0 + 1.0 + torch.randn(100, 5) * 0.1  # noise
    loader = DataLoader(TensorDataset(x, y_true), batch_size=20)
    q_hat = predictor.calibrate(loader)
    assert q_hat > 0
    assert predictor._calibrated is True
    # n_calib is the number of elements (samples * horizon * features),
    # not the number of samples. 100 samples * 5 features = 500 elements.
    assert predictor.n_calib == 500


def test_conformal_predictor_prediction():
    """Prediction should return point, lower, upper."""
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=10))
    x = torch.randn(50, 5)
    y_true = x * 2.0 + 1.0 + torch.randn(50, 5) * 0.1
    loader = DataLoader(TensorDataset(x, y_true), batch_size=10)
    predictor.calibrate(loader)
    # Predict
    x_test = torch.randn(10, 5)
    point, lower, upper = predictor.predict(x_test)
    assert point.shape == x_test.shape
    assert lower.shape == x_test.shape
    assert upper.shape == x_test.shape
    assert torch.all(lower <= point)
    assert torch.all(point <= upper)
    assert torch.allclose(upper - lower, torch.full_like(point, 2 * predictor.q_hat))


def test_conformal_predictor_coverage():
    """Empirical coverage should be close to 1 - alpha under exchangeability."""
    torch.manual_seed(42)
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=10))
    # Calibration set
    x_cal = torch.randn(200, 5)
    y_cal = x_cal * 2.0 + 1.0 + torch.randn(200, 5) * 0.1
    cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=50)
    predictor.calibrate(cal_loader)
    # Test set (same distribution -> exchangeability holds)
    x_test = torch.randn(500, 5)
    y_test = x_test * 2.0 + 1.0 + torch.randn(500, 5) * 0.1
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=50)
    coverage, mean_width, median_width = predictor.coverage_check(test_loader)
    # Coverage should be >= 0.85 (allowing for finite-sample variation)
    assert coverage >= 0.85, f"Coverage {coverage} < 0.85"


def test_conformal_predictor_state_dict():
    """state_dict and load_state_dict should round-trip correctly."""
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=10))
    x = torch.randn(50, 3)
    y = x * 2.0 + 1.0 + torch.randn(50, 3) * 0.1
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    predictor.calibrate(loader)
    state = predictor.state_dict()
    assert state["q_hat"] == predictor.q_hat
    # 50 samples * 3 features = 150 elements
    assert state["n_calib"] == 150
    # Round-trip
    model2 = DummyModel()
    predictor2 = ConformalPredictor(model2)
    predictor2.load_state_dict(state)
    assert predictor2.q_hat == predictor.q_hat
    assert predictor2._calibrated is True


def test_conformal_predictor_uncalibrated_raises():
    """predict before calibrate should raise RuntimeError."""
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=10))
    with pytest.raises(RuntimeError):
        predictor.predict(torch.randn(5, 3))


def test_conformal_predictor_small_calib_raises():
    """Calibration with fewer than min_calib samples should raise."""
    model = DummyModel()
    predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1, min_calib=100))
    x = torch.randn(10, 3)
    y = x * 2.0 + 1.0 + torch.randn(10, 3) * 0.1
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    with pytest.raises(RuntimeError):
        predictor.calibrate(loader)
