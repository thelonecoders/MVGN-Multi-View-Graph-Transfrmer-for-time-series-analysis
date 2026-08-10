"""
MVGT-Net: Conformal Uncertainty Quantification (Formula E)
==========================================================

Implements split-conformal prediction intervals for MVGT-Net point forecasts.

Theory (Section 6-16 of the thesis):
    Given a trained model f and a calibration set {(x_i, y_i)}_{i=1}^n,
    the conformal interval at a test point x is:
        C(x) = { y : |y - f(x)| <= q_hat }
    where q_hat is the (1 - alpha) empirical quantile of the absolute
    residuals {|y_i - f(x_i)|} on the calibration set.

Guarantee (Vovk et al. 2005; Lei et al. JASA 2018):
    Under exchangeability of (x_i, y_i) and the test point,
    P(y in C(x)) >= 1 - alpha.

Limitations (documented in Section 6-16):
    - The guarantee is marginal, not conditional.
    - Exchangeability breaks under distribution shift; for non-stationary
      TimeMMD domains (Economy, Climate), use EnbPI as alternative.
    - Coverage is on average over the calibration distribution, not
      per subgroup (per domain, per horizon).

References:
    Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World."
        Springer, 2005.
    Lei, Wasserman, et al. "Distribution-Free Prediction Bands for
        Non-parametric Regression." JASA 2018, pp. 1116-1127.
    Angelopoulos, Bates. "A Gentle Introduction to Conformal Prediction
        and Distribution-Free Uncertainty Quantification."
        Foundations and Trends in ML 2023, vol. 16, no. 2, pp. 494-591.
    Xu, Xie. "Conformal Prediction Interval for Dynamic Time-Series."
        ICML 2021, arXiv:2106.06214.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConformalConfig:
    """Configuration for the ConformalPredictor.

    Attributes:
        alpha: miscoverage rate (0.1 = 90% intervals, 0.05 = 95% intervals).
        method: 'split' for split conformal, 'enbpi' for Ensemble Batch
            Prediction Intervals (Xu & Xie 2021). Currently only 'split'
            is implemented; 'enbpi' is reserved for future work.
        min_calib: minimum calibration set size required to fit. Below
            this size, the predictor returns uncalibrated (infinite)
            intervals and logs a warning.
        device: torch device for model inference.
    """
    alpha: float = 0.1
    method: str = "split"
    min_calib: int = 30
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.method not in ("split", "enbpi"):
            raise ValueError(f"method must be 'split' or 'enbpi', got {self.method}")
        if self.min_calib < 1:
            raise ValueError(f"min_calib must be >= 1, got {self.min_calib}")


class ConformalPredictor(nn.Module):
    """Wraps a trained MVGT-Net model with conformal prediction intervals.

    Usage:
        model = MVGTNet(...)
        model.load_state_dict(torch.load("checkpoints/best.pt"))
        predictor = ConformalPredictor(model, ConformalConfig(alpha=0.1))
        predictor.calibrate(calib_loader)  # one-time calibration
        point, lower, upper = predictor.predict(x_batch)

    The calibration step computes q_hat once and caches it. The predict
    step returns the point forecast plus (point - q_hat, point + q_hat).
    The wrapper is non-trainable (does not modify the wrapped model).
    """

    def __init__(self, model: nn.Module, config: Optional[ConformalConfig] = None) -> None:
        super().__init__()
        self.model = model
        self.config = config or ConformalConfig()
        self.q_hat: Optional[float] = None
        self.n_calib: int = 0
        self._calibrated: bool = False

    @torch.no_grad()
    def calibrate(self, calib_loader: torch.utils.data.DataLoader) -> float:
        """Compute q_hat from the calibration set.

        Args:
            calib_loader: DataLoader yielding (inputs, targets) tuples.
                Inputs must be compatible with self.model.forward.
                Targets must have shape (batch, horizon) or (batch,).

        Returns:
            q_hat: the (1 - alpha) empirical quantile of absolute residuals.

        Raises:
            RuntimeError: if calibration set is smaller than config.min_calib.
        """
        self.model.eval()
        device = self.config.device
        residuals = []
        for batch in calib_loader:
            inputs, targets = batch[:2]
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = self.model(inputs)
            if preds.shape != targets.shape:
                # broadcast / squeeze if needed
                preds = preds.squeeze(-1) if preds.dim() == targets.dim() + 1 else preds
            abs_resid = (preds - targets).abs().flatten()
            residuals.append(abs_resid.cpu())
        if not residuals:
            raise RuntimeError("Calibration loader yielded no batches.")
        residuals_cat = torch.cat(residuals)
        self.n_calib = residuals_cat.numel()
        if self.n_calib < self.config.min_calib:
            raise RuntimeError(
                f"Calibration set too small: {self.n_calib} < {self.config.min_calib}. "
                f"Either increase calibration data or lower ConformalConfig.min_calib."
            )
        # Compute the (1 - alpha) quantile with the standard finite-sample
        # correction (n+1 in the denominator per Vovk et al. 2005, eq. 2.14).
        n = self.n_calib
        q_level = min(1.0, np.ceil((1 - self.config.alpha) * (n + 1)) / n)
        self.q_hat = float(torch.quantile(residuals_cat.to(torch.float32), q_level))
        self._calibrated = True
        return self.q_hat

    @torch.no_grad()
    def predict(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (point, lower, upper) for a batch of inputs.

        Args:
            inputs: model inputs, shape and type per MVGTNet.forward.

        Returns:
            Tuple of three tensors, each with the same shape as the model
            output. lower = point - q_hat, upper = point + q_hat.

        Raises:
            RuntimeError: if predict is called before calibrate.
        """
        if not self._calibrated or self.q_hat is None:
            raise RuntimeError(
                "ConformalPredictor.predict called before calibrate. "
                "Call predictor.calibrate(calib_loader) first."
            )
        self.model.eval()
        inputs = inputs.to(self.config.device)
        point = self.model(inputs)
        q = self.q_hat
        lower = point - q
        upper = point + q
        return point, lower, upper

    def coverage_check(
        self, test_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float]:
        """Empirical coverage and average interval width on a test set.

        This is a diagnostic: it does NOT validate the conformal guarantee
        (which holds marginally by construction if exchangeability holds),
        but it does check whether the test set violates exchangeability
        (which would show up as coverage < 1 - alpha).

        Args:
            test_loader: DataLoader yielding (inputs, targets).

        Returns:
            (empirical_coverage, mean_interval_width, median_interval_width)
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before coverage_check().")
        in_interval = 0
        total = 0
        widths = []
        for batch in test_loader:
            inputs, targets = batch[:2]
            point, lower, upper = self.predict(inputs)
            targets = targets.to(point.device)
            if point.shape != targets.shape:
                point = point.squeeze(-1) if point.dim() == targets.dim() + 1 else point
                lower = lower.squeeze(-1) if lower.dim() == targets.dim() + 1 else lower
                upper = upper.squeeze(-1) if upper.dim() == targets.dim() + 1 else upper
            in_int = ((targets >= lower) & (targets <= upper)).flatten()
            in_interval += int(in_int.sum().item())
            total += in_int.numel()
            widths.append((upper - lower).abs().flatten().cpu())
        coverage = in_interval / total if total > 0 else 0.0
        widths_cat = torch.cat(widths) if widths else torch.tensor([0.0])
        return coverage, float(widths_cat.mean()), float(widths_cat.median())

    def state_dict(self) -> dict:
        """Serialize the predictor state (model + q_hat + config)."""
        return {
            "model_state": self.model.state_dict(),
            "q_hat": self.q_hat,
            "n_calib": self.n_calib,
            "config": {
                "alpha": self.config.alpha,
                "method": self.config.method,
                "min_calib": self.config.min_calib,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Load the predictor state."""
        self.model.load_state_dict(state["model_state"])
        self.q_hat = state["q_hat"]
        self.n_calib = state["n_calib"]
        self._calibrated = self.q_hat is not None
        cfg = state["config"]
        self.config.alpha = cfg["alpha"]
        self.config.method = cfg["method"]
        self.config.min_calib = cfg["min_calib"]
