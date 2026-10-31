"""Glue between preprocessing and models: fit on a window, forecast a horizon.

Safeguards applied to every model output:
* non-finite forecasts are rejected (the model is reported as failed instead of plotting garbage);
* when the training target is non-negative, forecasts and interval bounds are clipped at zero;
* lower <= mean <= upper is enforced.
"""
import numpy as np
import pandas as pd

from forecasting.models import MODEL_REGISTRY
from preprocessing.cleaning import PrepConfig, apply_prep


def run_model(name, train, horizon, season, freq, level=0.95, nonneg=None):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    if train.isna().any():
        raise ValueError("Training window still contains missing values; choose a missing-value treatment.")
    y = train.values.astype(float)
    res = MODEL_REGISTRY[name](y, horizon, season, train.index, freq, level)
    mean, lo, hi = (np.asarray(res[k], float) for k in ("mean", "lower", "upper"))
    if not (np.all(np.isfinite(mean)) and len(mean) == horizon):
        raise ValueError(f"{name} produced non-finite or wrongly sized forecasts")
    lo = np.where(np.isfinite(lo), lo, mean); hi = np.where(np.isfinite(hi), hi, mean)
    if nonneg is None:
        nonneg = bool(np.nanmin(y) >= 0)
    if nonneg:
        mean, lo, hi = np.maximum(mean, 0), np.maximum(lo, 0), np.maximum(hi, 0)
    lo, hi = np.minimum(lo, mean), np.maximum(hi, mean)
    res = dict(res, mean=mean, lower=lo, upper=hi)
    res["meta"] = dict(res.get("meta", {}), clipped_at_zero=bool(nonneg))
    return res


def future_forecast(series, freq, season, cfg: PrepConfig, model_name, horizon, level=0.95):
    """Refit on ALL available (preprocessed) history and forecast `horizon` periods ahead."""
    train, log = apply_prep(series, cfg, season)
    res = run_model(model_name, train, horizon, season, freq, level)
    dates = pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]
    df = pd.DataFrame({"date": dates, "forecast": res["mean"], "lower": res["lower"], "upper": res["upper"]})
    return df, res["meta"], log
