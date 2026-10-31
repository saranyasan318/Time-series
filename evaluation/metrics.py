"""Forecast accuracy metrics. NaN pairs are ignored; MAPE ignores zero actuals.

MASE scales MAE by the in-sample MAE of a (seasonal) naive forecast, so it is comparable across series.
WAPE = sum|y-yhat| / sum|y|. R2 is only meaningful when the test window has variance (otherwise NaN).
"""
import numpy as np


def _clean(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    return y[m], p[m]


def mae(y, p): y, p = _clean(y, p); return float(np.mean(np.abs(y - p))) if len(y) else np.nan
def mse(y, p): y, p = _clean(y, p); return float(np.mean((y - p) ** 2)) if len(y) else np.nan
def rmse(y, p): v = mse(y, p); return float(np.sqrt(v)) if v == v else np.nan


def mape(y, p):
    """Percent; actuals equal to 0 are excluded (undefined). NaN if none remain."""
    y, p = _clean(y, p)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else np.nan


def smape(y, p):
    y, p = _clean(y, p)
    d = (np.abs(y) + np.abs(p)) / 2
    m = d != 0
    return float(np.mean(np.abs(y[m] - p[m]) / d[m]) * 100) if m.any() else np.nan


def wape(y, p):
    y, p = _clean(y, p)
    den = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / den * 100) if len(y) and den > 0 else np.nan


def mase(y, p, train, m=1):
    """MAE / in-sample MAE of the (seasonal) naive forecast on `train`. m=1 -> plain naive."""
    y, p = _clean(y, p)
    tr = np.asarray(train, float)
    tr = tr[~np.isnan(tr)]
    m = m if len(tr) > m + 1 else 1
    if len(tr) < 2 or not len(y):
        return np.nan
    scale = np.mean(np.abs(tr[m:] - tr[:-m]))
    return float(np.mean(np.abs(y - p)) / scale) if scale > 0 else np.nan


def r2(y, p):
    y, p = _clean(y, p)
    if len(y) < 2:
        return np.nan
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def all_metrics(y, p, train=None, m=1):
    out = {"MAE": mae(y, p), "MSE": mse(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "sMAPE": smape(y, p),
           "WAPE": wape(y, p), "MASE": mase(y, p, train, m) if train is not None else np.nan, "R2": r2(y, p)}
    return out
