"""
Evaluation Metrics
==================
Implements Eqs. 16-18 (MAE, RMSE, WAPE) plus 4 additional metrics
(MSE, MAPE, sMAPE, R^2) as proposed in Section 6-13 of the thesis.

All metrics are masked (ignore NaN/Inf entries) to handle missing data
in real-world time-series datasets like TimeMMD.
"""
import torch


def _mask_valid(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask of valid (non-Nan, non-Inf) entries."""
    mask = torch.isfinite(target) & torch.isfinite(pred)
    return mask


def masked_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (Eq. 16).

    Args:
        pred:   (B, S, N, C) predictions
        target: (B, S, N, C) ground truth

    Returns:
        scalar tensor
    """
    mask = _mask_valid(pred, target)
    diff = (pred - target).abs()
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    mask = _mask_valid(pred, target)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error (Eq. 17)."""
    return masked_mse(pred, target).sqrt()


def masked_wape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted Absolute Percentage Error (Eq. 18).

    WAPE = sum(|Y - Y_hat|) / sum(|Y|) * 100%
    """
    mask = _mask_valid(pred, target)
    numerator = ((pred - target).abs() * mask).sum()
    denominator = (target.abs() * mask).sum().clamp(min=1e-8)
    return 100.0 * numerator / denominator


def masked_mape(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error.

    MAPE = mean(|Y - Y_hat| / |Y|) * 100%
    """
    mask = _mask_valid(pred, target) & (target.abs() > eps)
    return 100.0 * ((pred - target).abs() / (target.abs() + eps) * mask).sum() / mask.sum().clamp(min=1)


def masked_smape(pred: torch.Tensor, target: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
    """Symmetric Mean Absolute Percentage Error.

    sMAPE = mean(2 * |Y - Y_hat| / (|Y| + |Y_hat|)) * 100%
    """
    mask = _mask_valid(pred, target)
    numerator = 2 * (pred - target).abs()
    denominator = (target.abs() + pred.abs()).clamp(min=eps)
    return 100.0 * (numerator / denominator * mask).sum() / mask.sum().clamp(min=1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R^2 (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot
    """
    mask = _mask_valid(pred, target)
    target_mean = (target * mask).sum() / mask.sum().clamp(min=1)
    ss_res = (((target - pred) ** 2) * mask).sum()
    ss_tot = (((target - target_mean) ** 2) * mask).sum().clamp(min=1e-8)
    return 1.0 - ss_res / ss_tot


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all 7 metrics at once."""
    return {
        "MAE": float(masked_mae(pred, target)),
        "MSE": float(masked_mse(pred, target)),
        "RMSE": float(masked_rmse(pred, target)),
        "WAPE": float(masked_wape(pred, target)),
        "MAPE": float(masked_mape(pred, target)),
        "sMAPE": float(masked_smape(pred, target)),
        "R2": float(r2_score(pred, target)),
    }
