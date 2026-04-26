from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PricePredictionModel:
    """Wrapper that keeps the target transform out of production callers."""

    model: Any
    residual_log_quantiles: dict[str, float]
    features: list[str]
    cat_features: list[str]
    target_transform: str = "log1p"

    def predict(self, X):
        raw_pred = self.model.predict(X)
        if self.target_transform == "log1p":
            return np.expm1(raw_pred).clip(min=0)
        return raw_pred

    def predict_interval(self, X, lower: str = "p10", upper: str = "p90"):
        raw_pred = self.model.predict(X)
        if self.target_transform == "log1p":
            lo = np.expm1(raw_pred + self.residual_log_quantiles[lower]).clip(min=0)
            mid = np.expm1(raw_pred).clip(min=0)
            hi = np.expm1(raw_pred + self.residual_log_quantiles[upper]).clip(min=0)
            return np.column_stack([lo, mid, hi])
        return np.column_stack([raw_pred, raw_pred, raw_pred])
