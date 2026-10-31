"""Exploratory time-series analysis. Numpy/pandas first; statsmodels is imported lazily."""
import numpy as np
import pandas as pd


def acf_values(x, nlags=24):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; x = x - x.mean()
    nlags = min(nlags, len(x) - 2)
    den = np.sum(x ** 2)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / den for k in range(1, nlags + 1)])


def acf_at(x, lag):
    a = acf_values(x, lag)
    return float(a[lag]) if len(a) > lag else 0.0


def pacf_values(x, nlags=24):
    """Partial autocorrelation via Levinson-Durbin on the sample ACF (no statsmodels needed)."""
    x = np.asarray(x, float)
    nlags = max(1, min(nlags, len(x) // 2 - 1))
    r = acf_values(x, nlags)
    pac = [1.0]
    phi_prev = np.array([])
    for k in range(1, nlags + 1):
        if k == 1:
            phi_kk = r[1]; phi = np.array([phi_kk])
        else:
            num = r[k] - np.sum(phi_prev * r[k - 1:0:-1])
            den = 1 - np.sum(phi_prev * r[1:k])
            phi_kk = num / den if den != 0 else 0.0
            phi = np.append(phi_prev - phi_kk * phi_prev[::-1], phi_kk)
        pac.append(float(phi_kk)); phi_prev = phi
    return np.array(pac)


def conf_band(n):
    return 1.96 / np.sqrt(n)


def summary_stats(s):
    s = s.dropna()
    return {"count": int(len(s)), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()),
            "max": float(s.max()), "start": s.index.min(), "end": s.index.max()}


def add_rolling(s, window):
    return pd.DataFrame({"value": s, "rolling_mean": s.rolling(window, min_periods=2).mean(),
                         "rolling_std": s.rolling(window, min_periods=2).std(), "pct_change": s.pct_change() * 100})


def trend_summary(s):
    """OLS slope per period and R2, classified as growth/decline/flat (|slope| < 0.1% of mean per period)."""
    y = s.dropna().values
    t = np.arange(len(y))
    slope, icpt = np.polyfit(t, y, 1)
    r2 = 1 - np.sum((y - (slope * t + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
    rel = slope / abs(y.mean()) * 100 if y.mean() else 0.0
    kind = "growth" if rel > 0.1 else "decline" if rel < -0.1 else "flat"
    return {"slope": float(slope), "pct_of_mean_per_period": float(rel), "r2": float(r2), "direction": kind}


def seasonal_profile(s, freq):
    """Mean by calendar position, as a ratio to the overall mean (1.0 = average)."""
    s = s.dropna()
    key = {"MS": s.index.month, "QS": s.index.quarter, "D": s.index.dayofweek, "W": s.index.isocalendar().week.values}[freq]
    g = s.groupby(key).mean()
    return g / s.mean()


def seasonality_strength(s, freq):
    p = seasonal_profile(s, freq)
    return float(p.max() - p.min())



def sudden_changes(s, z=3.0):
    s = s.dropna().sort_index()

    # Avoid misleading percentage changes near zero
    prev = s.shift(1)
    threshold = max(s.abs().median() * 0.01, 1e-8)

    pc = s.pct_change(fill_method=None)
    pc = pc[prev.abs() >= threshold].dropna()

    if len(pc) < 5 or pc.std() == 0:
        return pc.iloc[0:0]

    scores = (pc - pc.mean()) / pc.std()
    return pc[scores.abs() > z]



def dominant_cycle(s):
    """Dominant period (in observations) of the detrended series via FFT; None if weak."""
    y = s.dropna().values
    t = np.arange(len(y))
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    spec = np.abs(np.fft.rfft(y))[1:]
    if len(spec) < 3 or spec.sum() == 0: return None
    k = int(np.argmax(spec)) + 1
    share = float(spec[k - 1] ** 2 / np.sum(spec ** 2))
    return {"period": float(len(y) / k), "power_share": share}


_ADF_CRIT = [(-3.96, 0.001), (-3.43, 0.01), (-2.86, 0.05), (-2.57, 0.10), (-1.94, 0.30), (-0.50, 0.70), (0.9, 0.99)]


def _adf_basic(y, maxlag=None):
    """Augmented Dickey-Fuller (constant) with AIC lag choice; p-value interpolated from MacKinnon critical values
    (approximate). Used only when statsmodels is unavailable."""
    y = np.asarray(y, float)
    n = len(y)
    maxlag = maxlag if maxlag is not None else int(min(12, np.floor(12 * (n / 100) ** 0.25), n // 4))
    dy = np.diff(y)
    best = None
    for k in range(0, maxlag + 1):
        Y = dy[k:]
        cols = [y[k:-1] if k else y[:-1][k:], np.ones(len(Y))]
        cols[0] = y[k:-1]
        for j in range(1, k + 1):
            cols.append(dy[k - j:-j] if j else dy[k:])
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        sse = float(res @ res)
        nobs = len(Y)
        aic = nobs * np.log(sse / nobs + 1e-300) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            cov = sse / (nobs - X.shape[1]) * np.linalg.pinv(X.T @ X)
            best = (aic, float(beta[0] / np.sqrt(cov[0, 0])), k)
    stat = best[1]
    xs, ps = zip(*_ADF_CRIT)
    return stat, float(np.interp(stat, xs, ps)), best[2]


def _kpss_basic(y):
    """KPSS level-stationarity statistic with Bartlett-kernel long-run variance; p interpolated from critical values."""
    y = np.asarray(y, float); n = len(y)
    e = y - y.mean()
    S = np.cumsum(e)
    l = int(np.ceil(12 * (n / 100) ** 0.25))
    lrv = e @ e / n + 2 * sum((1 - j / (l + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, l + 1))
    stat = float(np.sum(S ** 2) / (n ** 2 * lrv))
    xs, ps = [0.119, 0.146, 0.347, 0.463, 0.574, 0.739], [0.99, 0.10, 0.10, 0.05, 0.025, 0.01]
    xs, ps = [0.0, 0.347, 0.463, 0.574, 0.739, 2.0], [0.10, 0.10, 0.05, 0.025, 0.01, 0.01]
    return stat, float(np.interp(stat, xs, ps))


def stationarity_tests(s):
    y = s.dropna().values
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        import warnings
        adf_p = float(adfuller(y, autolag="AIC")[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_p = float(kpss(y, regression="c", nlags="auto")[1])
        engine = "statsmodels"
    except ImportError:
        _, adf_p, _ = _adf_basic(y)
        _, kpss_p = _kpss_basic(y)
        engine = "built-in approximation (p-values interpolated from critical values)"
    if adf_p < 0.05 and kpss_p >= 0.05: verdict = "Stationary"
    elif adf_p >= 0.05 and kpss_p < 0.05: verdict = "Non-stationary (differencing recommended)"
    else: verdict = "Inconclusive (tests disagree; likely trend- or difference-stationary)"
    return {"adf_pvalue": adf_p, "kpss_pvalue": kpss_p, "verdict": verdict, "engine": engine}


def decompose(s, season, model="additive"):
    """Classical decomposition (centred moving-average trend, per-position seasonal means). NaN trend at the edges."""
    y = s.dropna()
    if season < 2 or len(y) < 2 * season:
        raise ValueError(f"Decomposition needs at least {2 * season} observations")
    mult = model == "multiplicative"
    if mult and (y <= 0).any():
        raise ValueError("Multiplicative decomposition needs strictly positive data")
    if season % 2 == 0:
        trend = y.rolling(season, center=True).mean().rolling(2).mean().shift(-1)
    else:
        trend = y.rolling(season, center=True).mean()
    detr = y / trend if mult else y - trend
    pos = np.arange(len(y)) % season
    seas_means = detr.groupby(pos).mean()
    seas_means = seas_means / seas_means.mean() if mult else seas_means - seas_means.mean()
    seasonal = pd.Series(seas_means.reindex(pos).values, index=y.index)
    resid = y / (trend * seasonal) if mult else y - trend - seasonal
    return pd.DataFrame({"observed": y, "trend": trend, "seasonal": seasonal, "residual": resid})


def strengths(dec):
    """Hyndman's trend/seasonal strength in [0,1] from a decomposition (additive)."""
    d = dec.dropna()
    if len(d) < 4:
        return {"trend": np.nan, "seasonal": np.nan}
    vr = np.var(d["residual"])
    ft = max(0.0, 1 - vr / np.var(d["trend"] + d["residual"])) if np.var(d["trend"] + d["residual"]) > 0 else 0.0
    fs = max(0.0, 1 - vr / np.var(d["seasonal"] + d["residual"])) if np.var(d["seasonal"] + d["residual"]) > 0 else 0.0
    return {"trend": float(ft), "seasonal": float(fs)}


def period_tables(s):
    """Mean by month, quarter, year plus year-over-year growth, for pattern charts and reports."""
    y = s.dropna()
    out = {"year": y.groupby(y.index.year).agg(["sum", "mean", "count"])}
    out["year"]["yoy_growth_%"] = out["year"]["sum"].pct_change() * 100
    out["quarter"] = y.groupby(y.index.quarter).mean()
    out["month"] = y.groupby(y.index.month).mean()
    out["month_year"] = y.groupby([y.index.year, y.index.month]).mean().unstack(0)
    return out


def characterise(s, freq, season):
    """One-stop description of the series used to choose models and write insights (all numbers computed)."""
    y = s.dropna()
    info = {"n": len(y), "trend": trend_summary(y), "cv_%": float(y.std() / abs(y.mean()) * 100) if y.mean() else np.nan}
    try:
        st = strengths(decompose(y, season)) if season > 1 else {"trend": np.nan, "seasonal": 0.0}
    except Exception:
        st = {"trend": np.nan, "seasonal": np.nan}
    info.update(trend_strength=st["trend"], seasonal_strength=st["seasonal"],
                seasonal_flag=bool(st["seasonal"] == st["seasonal"] and st["seasonal"] > 0.3))
    try:
        info["stationarity"] = stationarity_tests(y)
    except Exception as e:
        info["stationarity"] = {"verdict": f"unavailable ({e})"}
    info["cycle"] = dominant_cycle(y)
    return info
