"""Preprocessing: validation/clipping, context-aware imputation, feature engineering.

Kept as a scikit-learn transformer so it is saved *inside* the model pipeline —
the API can never apply different preprocessing than training did.

Context-aware imputation: from the *healthy* training rows we learn, for every
diagnostic sensor, a small linear model "what does a healthy car read here, given
RPM, load, throttle and speed?". A sensor the car does not report is filled with
that healthy expectation, so a missing PID can never look like a fault.

The same learned expectations replace hard-coded physics constants in the
engineered features (air flow vs. expected, oil pressure vs. expected), so the
features no longer copy the data generator's formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MIN_SENSORS_PRESENT, SENSORS, SENSOR_NAMES

_MINS = np.array([SENSORS[c]["min"] for c in SENSOR_NAMES], dtype=float)
_MAXS = np.array([SENSORS[c]["max"] for c in SENSOR_NAMES], dtype=float)

# Operating-point signals used to predict what the other sensors should read.
CONTEXT = ["engine_rpm", "engine_load", "throttle_pos", "vehicle_speed"]
_CTX_IDX = [SENSOR_NAMES.index(c) for c in CONTEXT]
_IDX = {name: i for i, name in enumerate(SENSOR_NAMES)}

ENGINEERED = [
    "total_fuel_trim",
    "maf_vs_expected",
    "oil_vs_expected",
    "coolant_minus_intake",
    "charging_deviation",
]


def to_frame(readings) -> pd.DataFrame:
    """Accept a dict, list of dicts or DataFrame and return the sensor columns.

    Absent sensors become NaN (imputed later by the pipeline) — many cars do not
    report every PID. Rows with fewer than MIN_SENSORS_PRESENT values are rejected.
    """
    if isinstance(readings, dict):
        readings = [readings]
    df = pd.DataFrame(readings).reindex(columns=SENSOR_NAMES)
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    present = df.notna().sum(axis=1)
    if (present < MIN_SENSORS_PRESENT).any():
        raise ValueError(
            f"Each reading needs at least {MIN_SENSORS_PRESENT} of {len(SENSOR_NAMES)} sensors; "
            f"got {int(present.min())}. Sensors: {', '.join(SENSOR_NAMES)}")
    return df


def _healthy_mask(y, n: int) -> np.ndarray:
    """Rows labelled healthy: 'normal' for class labels, all-zero for multi-label."""
    if y is None:
        return np.ones(n, dtype=bool)
    y = np.asarray(y)
    mask = (y.sum(axis=1) == 0) if y.ndim == 2 else (y == "normal")
    return mask if mask.sum() >= 50 else np.ones(n, dtype=bool)


def _design(ctx: np.ndarray) -> np.ndarray:
    rpm, load, throttle, speed = ctx.T
    return np.column_stack([np.ones(len(ctx)), rpm, load, rpm * load / 100, throttle, speed])


class OBDFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clip raw PIDs, impute missing ones as 'healthy for this context', add ratios."""

    def fit(self, X, y=None):
        X = np.array(X[SENSOR_NAMES] if isinstance(X, pd.DataFrame) else X, dtype=np.float64)
        healthy = X[_healthy_mask(y, len(X))]
        self.medians_ = np.nanmedian(healthy, axis=0)
        ctx = np.where(np.isnan(healthy[:, _CTX_IDX]), self.medians_[_CTX_IDX], healthy[:, _CTX_IDX])
        D = _design(np.clip(ctx, _MINS[_CTX_IDX], _MAXS[_CTX_IDX]))

        # One least-squares model per diagnostic sensor, fitted on rows where it is present.
        self.expect_coef_ = np.zeros((len(SENSOR_NAMES), D.shape[1]))
        for j, name in enumerate(SENSOR_NAMES):
            ok = ~np.isnan(healthy[:, j])
            if name in CONTEXT or ok.sum() < 50:
                self.expect_coef_[j, 0] = self.medians_[j]
            else:
                self.expect_coef_[j] = np.linalg.lstsq(D[ok], healthy[ok, j], rcond=None)[0]
        return self

    def expected(self, X: np.ndarray) -> np.ndarray:
        """Healthy expectation of every sensor given the operating point."""
        ctx = np.where(np.isnan(X[:, _CTX_IDX]), self.medians_[_CTX_IDX], X[:, _CTX_IDX])
        return _design(np.clip(ctx, _MINS[_CTX_IDX], _MAXS[_CTX_IDX])) @ self.expect_coef_.T

    def transform(self, X):
        # Plain NumPy: this runs several times per diagnosis, pandas overhead dominates.
        X = np.array(X[SENSOR_NAMES] if isinstance(X, pd.DataFrame) else X, dtype=np.float64)
        expected = np.clip(self.expected(X), _MINS, _MAXS)
        X = np.clip(np.where(np.isnan(X), expected, X), _MINS, _MAXS)

        c = {name: X[:, i] for name, i in _IDX.items()}
        engineered = np.column_stack([
            c["short_fuel_trim"] + c["long_fuel_trim"],
            c["maf_rate"] / np.maximum(expected[:, _IDX["maf_rate"]], 0.5),
            c["oil_pressure"] / np.maximum(expected[:, _IDX["oil_pressure"]], 20),
            c["coolant_temp"] - c["intake_air_temp"],
            np.abs(c["battery_voltage"] - self.medians_[_IDX["battery_voltage"]]),
        ])
        return np.hstack([X, engineered])

    def get_feature_names_out(self, input_features=None):
        return np.array(SENSOR_NAMES + ENGINEERED)
