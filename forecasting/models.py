"""Forecasting models.

Every model has the signature  f(y, h, season, dates, freq, level) -> {'mean','lower','upper','meta'}
where y is a clean 1-D numpy array (no NaN). Models that need statsmodels / xgboost / prophet / pmdarima import
them lazily and are reported as *unavailable* (never silently dropped) when the package is missing.
Non-negativity is enforced centrally in forecasting.pipeline.run_model.
"""
import importlib.util
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import ML_TREND_DAMPING, MA_WINDOW, MULTI_SEASONS, RANDOM_STATE
from forecasting.ets import ETSResult
from forecasting.features import build_supervised, choose_lags, row_features


def _has(pkg):
    return importlib.util.find_spec(pkg) is not None


def _z(level):
    return float(norm.ppf(0.5 + level / 2))


def _bounds(mean, sigma, level, scale=None):
    k = np.sqrt(np.arange(1, len(mean) + 1)) if scale is None else np.asarray(scale, float)
    z = _z(level)
    return mean - z * sigma * k, mean + z * sigma * k


def _out(mean, sigma, level, meta=None, scale=None):
    mean = np.asarray(mean, float)
    lo, hi = _bounds(mean, sigma if np.isfinite(sigma) else 0.0, level, scale)
    return {"mean": mean, "lower": lo, "upper": hi, "meta": meta or {}}


def _need(y, n, name):
    if len(y) < n:
        raise ValueError(f"{name} needs at least {n} observations (have {len(y)})")


# ----------------------------------------------------------------------------- baselines
def naive(y, h, season, dates, freq, level):
    _need(y, 3, "Naive")
    return _out(np.repeat(y[-1], h), np.std(np.diff(y)), level)


def seasonal_naive(y, h, season, dates, freq, level):
    m = max(season, 1)
    _need(y, m + 2, "Seasonal naive")
    mean = np.array([y[-m + (i % m)] for i in range(h)])
    sigma = np.std(y[m:] - y[:-m])
    return _out(mean, sigma, level, scale=np.sqrt(np.ceil(np.arange(1, h + 1) / m)))


def drift(y, h, season, dates, freq, level):
    _need(y, 6, "Drift")
    slope = (y[-1] - y[0]) / (len(y) - 1)
    fitted = y[:-1] + slope
    sigma = np.std(y[1:] - fitted) * np.sqrt(len(y) / max(len(y) - 1, 1))
    steps = np.arange(1, h + 1)
    return _out(y[-1] + slope * steps, sigma, level, {"drift_per_period": float(slope)})


def simple_moving_average(y, h, season, dates, freq, level):
    w = MA_WINDOW[freq]
    _need(y, w + 3, "Moving average")
    errs = np.array([y[i] - y[i - w:i].mean() for i in range(w, len(y))])
    return _out(np.repeat(y[-w:].mean(), h), np.std(errs), level, {"window": w})


def rolling_average(y, h, season, dates, freq, level):
    w = MA_WINDOW[freq]
    _need(y, w + 3, "Rolling average")
    hist, out = list(y), []
    for _ in range(h):
        out.append(np.mean(hist[-w:])); hist.append(out[-1])
    errs = np.array([y[i] - y[i - w:i].mean() for i in range(w, len(y))])
    return _out(out, np.std(errs), level, {"window": w})


# ----------------------------------------------------------------------------- exponential smoothing
def _ets(y, h, season, level, trend, damped, seasonal, name):
    m = max(season, 1)
    if seasonal:
        if m < 2:
            raise ValueError(f"{name} needs a seasonal period >= 2")
        _need(y, 2 * m + 2, name)
    else:
        _need(y, 6, name)
    r = ETSResult(y, "add" if trend else None, damped, seasonal, m).fit()
    mean = r.forecast(h)
    sd = r.std_path(h)
    z = _z(level)
    meta = {"alpha": float(r.params[0]), "sigma": r.sigma}
    return {"mean": mean, "lower": mean - z * sd, "upper": mean + z * sd, "meta": meta}


def ses(y, h, season, dates, freq, level): return _ets(y, h, season, level, False, False, None, "SES")
def holt(y, h, season, dates, freq, level): return _ets(y, h, season, level, True, False, None, "Holt")
def damped_holt(y, h, season, dates, freq, level): return _ets(y, h, season, level, True, True, None, "Damped Holt")
def holt_winters(y, h, season, dates, freq, level): return _ets(y, h, season, level, True, False, "add", "Holt-Winters")


def holt_winters_damped(y, h, season, dates, freq, level):
    return _ets(y, h, season, level, True, True, "add", "Holt-Winters (damped)")


def holt_winters_mul(y, h, season, dates, freq, level):
    if np.any(y <= 0):
        raise ValueError("Multiplicative Holt-Winters requires strictly positive data")
    return _ets(y, h, season, level, True, False, "mul", "Holt-Winters (multiplicative)")


def theta(y, h, season, dates, freq, level):
    """Theta method (Assimakopoulos & Nikolopoulos): SES + half the linear drift, on a seasonally adjusted series."""
    _need(y, 8, "Theta")
    m = max(season, 1)
    n = len(y)
    idx = np.zeros(max(m, 1))
    adj = y
    if m > 1 and n >= 2 * m:   # classical additive seasonal indices from de-trended data
        t = np.arange(n)
        resid = y - np.polyval(np.polyfit(t, y, 1), t)
        idx = np.array([resid[k::m].mean() for k in range(m)]); idx -= idx.mean()
        adj = y - idx[np.arange(n) % m]
    r = ETSResult(adj, None, False, None, 1).fit()
    alpha = float(r.params[0])
    b0 = np.polyfit(np.arange(n), adj, 1)[0]
    steps = np.arange(1, h + 1)
    mean = r.forecast(h) + 0.5 * b0 * ((steps - 1) + 1 / alpha - ((1 - alpha) ** n) / alpha)
    if m > 1 and n >= 2 * m:
        mean = mean + idx[(n + steps - 1) % m]
    z, sd = _z(level), r.std_path(h)
    return {"mean": mean, "lower": mean - z * sd, "upper": mean + z * sd, "meta": {"alpha": alpha, "drift": float(b0 / 2)}}


# ----------------------------------------------------------------------------- regression with Fourier terms
def _fourier(t, periods, K):
    cols = []
    for P, k in zip(periods, K):
        for j in range(1, k + 1):
            cols += [np.sin(2 * np.pi * j * t / P), np.cos(2 * np.pi * j * t / P)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def harmonic_regression(y, h, season, dates, freq, level):
    """Linear trend + Fourier terms for every calendar period the history covers twice (multi-seasonal)."""
    n = len(y)
    _need(y, 12, "Harmonic regression")
    # two full cycles for short periods; 1.5 cycles suffice for long ones (e.g. a yearly cycle in daily data)
    periods = [p for p in MULTI_SEASONS[freq] if n >= (2 if p <= 60 else 1.5) * p]
    if not periods:
        raise ValueError("History is shorter than two cycles of every seasonal period")
    K = [min(int(p // 2), 3 if p <= 12 else 4) for p in periods]
    t = np.arange(n + h, dtype=float)
    X = np.column_stack([np.ones(n + h), t, _fourier(t, periods, K)])
    from sklearn.linear_model import Ridge
    mdl = Ridge(alpha=1e-6, fit_intercept=False).fit(X[:n], y)
    fit = mdl.predict(X[:n])
    sigma = float(np.std(y - fit, ddof=min(X.shape[1], n - 1)))
    mean = mdl.predict(X[n:])
    return _out(mean, sigma, level, {"periods": periods, "fourier_terms": K}, scale=np.ones(h) * np.sqrt(1 + X.shape[1] / n))


# ----------------------------------------------------------------------------- ARIMA family (statsmodels)
def choose_d(y, max_d=2):
    """Differencing order via repeated ADF tests (p > 0.05 -> difference again)."""
    from statsmodels.tsa.stattools import adfuller
    d, x = 0, np.asarray(y, float)
    while d < max_d and len(x) > 10 and adfuller(x, autolag="AIC")[1] > 0.05:
        x, d = np.diff(x), d + 1
    return d


def arima(y, h, season, dates, freq, level):
    from statsmodels.tsa.arima.model import ARIMA
    _need(y, 12, "ARIMA")
    d = choose_d(y)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(3):
            for q in range(3):
                for trend in (["t", "n"] if d == 1 else ["c"] if d == 0 else ["n"]):
                    try:
                        fit = ARIMA(y, order=(p, d, q), trend=trend).fit()
                    except Exception:
                        continue
                    if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                        best = (fit.aic, (p, d, q), fit)
                    break
        if best is None:
            raise ValueError("No ARIMA configuration could be fitted")
        fc = best[2].get_forecast(h)
        ci = np.asarray(fc.conf_int(alpha=1 - level))
        mean = np.asarray(fc.predicted_mean)
    return {"mean": mean, "lower": ci[:, 0], "upper": ci[:, 1], "meta": {"order": best[1], "aic": float(best[0])}}


def sarima(y, h, season, dates, freq, level):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from exploratory_analysis.eda import acf_at
    m = season
    if m < 2 or m > 24:
        raise ValueError("SARIMA needs a seasonal period between 2 and 24")
    _need(y, 2 * m + 4, "SARIMA")
    D = 1 if (len(y) >= 3 * m and acf_at(y, m) > 0.5) else 0
    d = choose_d(y[m:] - y[:-m] if D else y, max_d=1)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for P, Q in [(1, 0), (0, 1), (1, 1)]:
                try:
                    fit = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, m),
                                  trend="c" if d + D == 0 else None).fit(disp=False, maxiter=100)
                except Exception:
                    continue
                if np.isfinite(fit.aic) and (best is None or fit.aic < best[0]):
                    best = (fit.aic, (p, d, q), (P, D, Q, m), fit)
        if best is None:
            raise ValueError("No SARIMA configuration could be fitted")
        fc = best[3].get_forecast(h)
        ci = np.asarray(fc.conf_int(alpha=1 - level))
        mean = np.asarray(fc.predicted_mean)
    return {"mean": mean, "lower": ci[:, 0], "upper": ci[:, 1],
            "meta": {"order": best[1], "seasonal_order": best[2], "aic": float(best[0])}}


def auto_arima(y, h, season, dates, freq, level):
    import pmdarima as pm
    _need(y, 24, "Auto ARIMA")
    m = season if 2 <= season <= 24 and len(y) >= 2 * season + 4 else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl = pm.auto_arima(y, seasonal=m > 1, m=m, suppress_warnings=True, error_action="ignore", stepwise=True,
                            max_p=3, max_q=3, max_P=2, max_Q=2, information_criterion="aic")
        mean, ci = mdl.predict(h, return_conf_int=True, alpha=1 - level)
    return {"mean": np.asarray(mean), "lower": ci[:, 0], "upper": ci[:, 1],
            "meta": {"order": mdl.order, "seasonal_order": mdl.seasonal_order}}


# ----------------------------------------------------------------------------- machine learning (recursive, leak-free)
def _ml(y, h, season, dates, freq, level, factory, name):
    """Remove a linear trend, learn the remainder from lag/rolling/calendar features, forecast recursively
    and add a damped trend back. Prediction spread comes from a chronological one-step hold-out."""
    n = len(y)
    lags = choose_lags(freq, n)
    windows = sorted({w for w in (3, min(max(season, 4), max(lags))) if w <= max(lags)}) or [2]
    _need(y, max(lags + windows) + 8, name)
    t = np.arange(n)
    slope, icpt = np.polyfit(t, y, 1)
    resid = y - (slope * t + icpt)
    dates = pd.DatetimeIndex(dates)
    X, tgt = build_supervised(resid, dates, lags, windows)
    model = factory().fit(X, tgt)
    k = int(0.75 * len(X))
    if k >= 5 and len(X) - k >= 3:
        val = factory().fit(X[:k], tgt[:k])
        sigma = float(np.std(tgt[k:] - val.predict(X[k:])))
    else:
        sigma = float(np.std(tgt - model.predict(X)))
    future = pd.date_range(dates[-1], periods=h + 1, freq=freq)[1:]
    hist, pred = list(resid), []
    for i in range(h):
        p = float(model.predict([row_features(np.array(hist), future[i], lags, windows)])[0])
        hist.append(p); pred.append(p)
    steps = np.arange(1, h + 1)
    phi = ML_TREND_DAMPING
    damped_trend = slope * np.cumsum(phi ** steps)
    mean = np.array(pred) + (slope * (n - 1) + icpt) + damped_trend
    return _out(mean, sigma, level, {"lags": lags, "windows": windows, "trend_slope": float(slope)})


def random_forest(y, h, season, dates, freq, level):
    from sklearn.ensemble import RandomForestRegressor
    return _ml(y, h, season, dates, freq, level,
               lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
               "Random Forest")


def gradient_boosting(y, h, season, dates, freq, level):
    from sklearn.ensemble import GradientBoostingRegressor
    return _ml(y, h, season, dates, freq, level,
               lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8,
                                                 random_state=RANDOM_STATE), "Gradient Boosting")


def extra_trees(y, h, season, dates, freq, level):
    from sklearn.ensemble import ExtraTreesRegressor
    return _ml(y, h, season, dates, freq, level,
               lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
               "Extra Trees")


def xgboost_model(y, h, season, dates, freq, level):
    from xgboost import XGBRegressor
    return _ml(y, h, season, dates, freq, level,
               lambda: XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8,
                                    random_state=RANDOM_STATE, n_jobs=1, verbosity=0), "XGBoost")


def prophet_model(y, h, season, dates, freq, level):
    from prophet import Prophet
    _need(y, 24, "Prophet")
    df = pd.DataFrame({"ds": pd.DatetimeIndex(dates), "y": y})
    m = Prophet(interval_width=level, weekly_seasonality=freq == "D", yearly_seasonality=len(y) >= 24 or freq == "D",
                daily_seasonality=False)
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    m.fit(df)
    fut = m.make_future_dataframe(periods=h, freq=freq).tail(h)
    p = m.predict(fut)
    return {"mean": p["yhat"].values, "lower": p["yhat_lower"].values, "upper": p["yhat_upper"].values, "meta": {}}


# ----------------------------------------------------------------------------- registry
# name -> (function, required packages, family)
_SPEC = {
    "Naive": (naive, [], "Baseline"),
    "Seasonal Naive": (seasonal_naive, [], "Baseline"),
    "Drift": (drift, [], "Baseline"),
    "Simple Moving Average": (simple_moving_average, [], "Baseline"),
    "Rolling Average (recursive)": (rolling_average, [], "Baseline"),
    "Simple Exp. Smoothing": (ses, [], "Exponential smoothing"),
    "Holt Linear Trend": (holt, [], "Exponential smoothing"),
    "Damped Holt": (damped_holt, [], "Exponential smoothing"),
    "Holt-Winters": (holt_winters, [], "Exponential smoothing"),
    "Holt-Winters (damped)": (holt_winters_damped, [], "Exponential smoothing"),
    "Holt-Winters (multiplicative)": (holt_winters_mul, [], "Exponential smoothing"),
    "Theta": (theta, [], "Exponential smoothing"),
    "Harmonic Regression": (harmonic_regression, [], "Regression"),
    "ARIMA": (arima, ["statsmodels"], "ARIMA"),
    "SARIMA": (sarima, ["statsmodels"], "ARIMA"),
    "Auto ARIMA": (auto_arima, ["pmdarima"], "ARIMA"),
    "Random Forest": (random_forest, [], "Machine learning"),
    "Gradient Boosting": (gradient_boosting, [], "Machine learning"),
    "Extra Trees": (extra_trees, [], "Machine learning"),
    "XGBoost": (xgboost_model, ["xgboost"], "Machine learning"),
    "Prophet": (prophet_model, ["prophet"], "Machine learning"),
}
MODEL_REGISTRY = {k: v[0] for k, v in _SPEC.items()}
MODEL_FAMILY = {k: v[2] for k, v in _SPEC.items()}
BASELINES = ["Naive", "Seasonal Naive", "Drift"]


def model_availability():
    """{model: (available: bool, reason: str)} - missing optional packages are reported, not hidden."""
    out = {}
    for name, (_, pkgs, _) in _SPEC.items():
        missing = [p for p in pkgs if not _has(p)]
        out[name] = (not missing, "" if not missing else f"requires {', '.join(missing)} (pip install {' '.join(missing)})")
    return out


def available_models():
    return [k for k, (ok, _) in model_availability().items() if ok]
