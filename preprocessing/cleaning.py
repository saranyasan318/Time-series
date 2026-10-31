"""Validation, regularisation, missing-value and outlier handling. The raw DataFrame is never modified."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DEFAULT_MISSING, MIN_OBSERVATIONS, MISSING_METHODS, OUTLIER_METHODS, SEASON_LENGTH


@dataclass
class PrepConfig:
    missing_method: str = DEFAULT_MISSING
    outlier_method: str = OUTLIER_METHODS[0]
    outlier_threshold: float = 3.5      # robust z-score (median/MAD) above which a point is flagged


def detect_frequency(dates):
    """Return 'D','W','MS','QS' from the median spacing of timestamps, or None."""
    d = pd.Series(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if len(d) < 3:
        return None
    med = d.diff().dropna().dt.total_seconds().median() / 86400
    if med <= 1.5: return "D"
    if 5.5 <= med <= 8.5: return "W"
    if 27 <= med <= 32: return "MS"
    if 85 <= med <= 95: return "QS"
    return None


def parse_dates(df, date_col, fmt=None):
    """Parse a column to datetimes. `fmt` is a strptime format, or None to auto-detect. Returns (dates, n_invalid)."""
    col = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(col):
        parsed = col
    elif fmt:
        parsed = pd.to_datetime(col, format=fmt, errors="coerce")
    else:
        parsed = pd.to_datetime(col, errors="coerce", format="mixed") if not pd.api.types.is_numeric_dtype(col) \
            else pd.to_datetime(col, errors="coerce")
    return parsed, int(parsed.isna().sum())


def detect_date_format(values, candidates=None):
    """Best strptime format for a column of date strings (highest parse rate; ties go to the first candidate).
    Returns (format or None, parse_rate, ambiguous_day_month). Ambiguous = both DD/MM and MM/DD parse everything."""
    from config import DATE_FORMATS
    v = pd.Series(values).dropna().astype(str)
    if v.empty:
        return None, 0.0, False
    cands = [f for f in (candidates or DATE_FORMATS.values()) if f]
    rates = {f: float(pd.to_datetime(v, format=f, errors="coerce").notna().mean()) for f in cands}
    best = max(rates, key=lambda f: (rates[f], -cands.index(f)))
    ambiguous = rates.get("%d/%m/%Y", 0) == 1.0 and rates.get("%m/%d/%Y", 0) == 1.0
    return (best if rates[best] > 0 else None), rates[best], ambiguous


def guess_date_col(df):
    """Date column: name hint first, then any non-numeric column that mostly parses as dates."""
    cols = list(df.columns)
    hit = next((c for c in cols if any(k in str(c).lower() for k in ("date", "time", "month", "day", "period"))), None)
    if hit is not None:
        return hit
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]) and pd.to_datetime(df[c], errors="coerce", format="mixed").notna().mean() > .8:
            return c
    return cols[0]


def guess_target_col(df, date_col):
    """Numeric target: prefer business-like names, otherwise the numeric column with the most variance in relative terms."""
    num = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return None
    pref = ("sales", "revenue", "demand", "amount", "value", "quantity", "target", "total", "price")
    for k in pref:
        for c in num:
            if k in str(c).lower():
                return c
    return max(num, key=lambda c: df[c].notna().sum())


def validate_dataset(df, date_col, target_col, group_col=None, freq=None, min_obs=None, date_format=None):
    """Return {'errors','warnings','info'}; errors block modelling, warnings do not."""
    errors, warnings, info = [], [], {}
    if date_col not in df or target_col not in df:
        return {"errors": ["Selected columns not found."], "warnings": [], "info": {}}
    dates, bad_dates = parse_dates(df, date_col, date_format)
    target = pd.to_numeric(df[target_col], errors="coerce")
    non_numeric = int((target.isna() & df[target_col].notna()).sum())
    missing_target = int(df[target_col].isna().sum())
    info.update(rows=len(df), invalid_dates=bad_dates, non_numeric_target=non_numeric, missing_target=missing_target)
    if bad_dates == len(df): errors.append("No valid dates: the date column cannot be parsed.")
    elif bad_dates: warnings.append(f"{bad_dates} rows have invalid dates and will be dropped.")
    if non_numeric == len(df) - missing_target and non_numeric > 0: errors.append("Target column is not numeric.")
    elif non_numeric: warnings.append(f"{non_numeric} non-numeric target values will be treated as missing.")
    if missing_target: warnings.append(f"{missing_target} missing target values.")
    if errors: return {"errors": errors, "warnings": warnings, "info": info}
    keys = [dates] + ([df[group_col]] if group_col else [])
    valid = dates.notna()
    dup = int(pd.concat(keys, axis=1)[valid].duplicated().sum())
    info["duplicate_timestamps"] = dup
    if dup: warnings.append(f"{dup} duplicate timestamps will be aggregated.")
    det = detect_frequency(dates[valid])
    info["detected_frequency"] = det
    use = freq or det
    if det is None: warnings.append("Frequency could not be detected; choose one manually (data will be resampled).")
    elif freq and freq != det: warnings.append(f"Chosen frequency {freq} differs from detected {det}; data will be resampled.")
    u = pd.DatetimeIndex(dates[valid].unique()).sort_values()
    if len(u) > 3:
        gaps = pd.Series(u).diff().dropna().dt.days
        share = float((gaps != gaps.mode().iloc[0]).mean())
        info["irregular_share"] = share
        if share > 0.10 and det in ("D", "W"): warnings.append(f"{share:.0%} of intervals are irregular.")
    s = pd.Series(target.values, index=dates)[valid.values]
    if s.nunique(dropna=True) <= 1: errors.append("Target is constant; nothing to forecast.")
    if (s.dropna() < 0).any(): warnings.append("Negative target values found; MAPE may be unreliable.")
    n_min = min_obs or MIN_OBSERVATIONS.get(use, 24)
    n_obs = len(u)
    info["distinct_timestamps"] = n_obs
    if n_obs < n_min: errors.append(f"Insufficient history: {n_obs} timestamps, at least {n_min} required.")
    return {"errors": errors, "warnings": warnings, "info": info}


def prepare_series(df, date_col, target_col, freq, group_col=None, group_value=None, agg="sum", date_format=None):
    """Parse, sort, de-duplicate, filter category and resample to a regular grid (gaps stay NaN)."""
    d = df[[c for c in {date_col, target_col, group_col} if c]].copy()
    d[date_col] = parse_dates(d, date_col, date_format)[0]
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if group_col and group_value not in (None, "ALL"):
        d = d[d[group_col].astype(str) == str(group_value)]
    d = d.sort_values(date_col)
    log = [f"Rows after dropping invalid dates/filtering: {len(d)}"]
    g = d.groupby(date_col)[target_col]
    if agg == "sum": s = g.sum(min_count=1)
    elif agg == "mean": s = g.mean()
    else: s = g.last()
    log.append(f"Duplicate timestamps aggregated with '{agg}'")
    rs = s.resample(freq)
    s = rs.sum(min_count=1) if agg == "sum" else (rs.mean() if agg == "mean" else rs.last())
    s.name = target_col
    log.append(f"Resampled to '{freq}': {len(s)} periods, {int(s.isna().sum())} missing")
    return s, log


def missing_timestamps(df, date_col, freq, date_format=None):
    d = parse_dates(df, date_col, date_format)[0].dropna()
    full = pd.date_range(d.min().to_period("M").to_timestamp() if freq == "MS" else d.min(), d.max(), freq=freq)
    return full.difference(pd.DatetimeIndex(d.unique()))


def _gap_runs(s):
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    return isna.groupby(grp).transform("sum") * isna


def treat_missing(s, method, season=1):
    """Fill only where the chosen method is defensible. Returns (series, n_filled)."""
    before = s.isna().sum()
    out = s.copy()
    if method == MISSING_METHODS[1]:
        fill = _gap_runs(s) <= 2
        interp = s.interpolate(limit_area="inside")
        m = s.isna() & fill
        out[m] = interp[m]
    elif method == MISSING_METHODS[2] and season > 1:
        pos = np.arange(len(s)) % season
        means = s.groupby(pos).mean()
        for i in np.where(s.isna().values)[0]:
            out.iloc[i] = means.get(pos[i], np.nan)
    elif method == MISSING_METHODS[3]:
        out = out.fillna(s.median())
    elif method == MISSING_METHODS[4]:
        out = out.ffill(limit=1)
    elif method == MISSING_METHODS[5]:
        out = s.interpolate(limit_area="inside")
    return out, int(before - out.isna().sum())


def _centered_ma_trend(s, m):
    """Classical centred moving-average trend (2xm for even m); edges extrapolated linearly from the nearest m points."""
    v = s.astype(float)
    if m % 2 == 0:
        ma = v.rolling(m, center=True).mean().rolling(2).mean().shift(-1)
    else:
        ma = v.rolling(m, center=True).mean()
    arr = ma.to_numpy(dtype=float, copy=True)
    ok = ~np.isnan(arr)
    if ok.sum() >= 2:
        idx = np.arange(len(v))
        good = idx[ok]
        k = min(m, len(good))
        for edge, sel in (("l", good[:k]), ("r", good[-k:])):
            slope, icpt = np.polyfit(sel, arr[sel], 1)
            miss = idx[(idx < good[0])] if edge == "l" else idx[(idx > good[-1])]
            arr[miss] = slope * miss + icpt
    return pd.Series(arr, index=s.index)


def detect_outliers(s, window=None, thresh=3.5, season=1):
    """Robust z-score (median/MAD) of residuals. With season > 1 (and >= 2 cycles) the per-position seasonal
    component is removed first so regular seasonal peaks are not mistaken for outliers."""
    orig = pd.Series(s)
    s = orig.interpolate(limit_area=None).bfill().ffill()   # gaps must not blank out the trend estimate
    if s.isna().all():
        return pd.Series(False, index=orig.index)
    w = max(5, min(window or 7, max(5, len(s) // 2)))
    if season > 1 and len(s) >= 2 * season:
        trend = _centered_ma_trend(s, season)
        dev = s - trend
        pos = np.arange(len(s)) % season
        resid = dev - dev.groupby(pos).transform("median")
    else:
        resid = s - s.rolling(w, center=True, min_periods=3).median()
    c = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - c))
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(resid) or 1.0
    z = 0.6745 * (resid - c) / mad
    return (z.abs() > thresh) & orig.notna()


def treat_outliers(s, mask, method):
    out = s.copy()
    if method == OUTLIER_METHODS[1]:
        q1, q3 = s.quantile(.25), s.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        out[mask] = s[mask].clip(lo, hi)
    elif method == OUTLIER_METHODS[2]:
        med = s.rolling(5, center=True, min_periods=2).median()
        out[mask] = med[mask]
    return out


def apply_prep(s, cfg: PrepConfig, season=1):
    """Apply preprocessing to ONE window of data (call on training data only to avoid leakage)."""
    log = []
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None: return s, ["Series is entirely missing"]
    s = s.loc[first:last]
    mask = detect_outliers(s, window=season if season > 1 else 7, thresh=cfg.outlier_threshold, season=season)
    log.append(f"Outliers flagged: {int(mask.sum())} at |robust z| > {cfg.outlier_threshold} (treatment: {cfg.outlier_method})")
    s = treat_outliers(s, mask, cfg.outlier_method)
    s, n = treat_missing(s, cfg.missing_method, season)
    log.append(f"Missing values filled: {n}; remaining missing: {int(s.isna().sum())}")
    return s, log


def consistency_checks(s):
    return {"monotonic_index": bool(s.index.is_monotonic_increasing), "unique_index": bool(s.index.is_unique),
            "regular_grid": bool(len(s) < 3 or s.index.to_series().diff().dropna().nunique() <= 3),
            "n_missing": int(s.isna().sum()), "n_negative": int((s < 0).sum())}


def difference(s, d=0, D=0, m=1):
    """Apply D seasonal differences (lag m) then d ordinary differences. Returns a new series; input untouched."""
    out = s.copy()
    for _ in range(D):
        out = out.diff(m)
    for _ in range(d):
        out = out.diff()
    return out.dropna()


def suggest_differencing(s, season=1, max_d=2):
    """Suggest (d, D) from ADF tests when statsmodels is present, else from a variance-reduction heuristic.
    Returns {'d','D','method'}."""
    y = s.dropna().values.astype(float)
    try:
        from statsmodels.tsa.stattools import adfuller
        D = 0
        if season > 1 and len(y) >= 3 * season:
            from exploratory_analysis.eda import acf_at
            D = 1 if acf_at(y, season) > 0.5 else 0
        x = y[season:] - y[:-season] if D else y
        d = 0
        while d < max_d and len(x) > 10 and adfuller(x, autolag="AIC")[1] > 0.05:
            x, d = np.diff(x), d + 1
        return {"d": d, "D": D, "method": "ADF test"}
    except ImportError:
        d, x = 0, y
        while d < max_d and len(x) > 10 and np.var(np.diff(x)) < 0.8 * np.var(x):
            x, d = np.diff(x), d + 1
        return {"d": d, "D": 0, "method": "variance heuristic (install statsmodels for ADF-based choice)"}


def interval_profile(dates):
    """Spacing statistics of timestamps: modal gap in days, share of off-mode gaps, largest gap."""
    u = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if len(u) < 3:
        return {"modal_gap_days": None, "irregular_share": 0.0, "max_gap_days": None, "is_regular": True}
    gaps = pd.Series(u).diff().dropna().dt.days
    mode = float(gaps.mode().iloc[0])
    tol = 3 if mode >= 28 else 0           # months differ by up to 3 days
    share = float((np.abs(gaps - mode) > tol).mean())
    return {"modal_gap_days": mode, "irregular_share": share, "max_gap_days": float(gaps.max()), "is_regular": share <= 0.05}


def data_quality_report(df, date_col, target_col, freq=None, group_col=None, fmt=None, outlier_thresh=3.5, group_value=None):
    """Structured, numeric data-quality summary. Score is 100 minus capped penalties (documented in 'penalties')."""
    if group_col and group_value not in (None, "ALL"):
        df = df[df[group_col].astype(str) == str(group_value)]
    dates, bad = parse_dates(df, date_col, fmt)
    tgt = pd.to_numeric(df[target_col], errors="coerce")
    n = len(df)
    valid = dates.notna()
    keys = [dates] + ([df[group_col]] if group_col else [])
    dup = int(pd.concat(keys, axis=1)[valid].duplicated().sum())
    det = detect_frequency(dates[valid]) if valid.any() else None
    use = freq or det
    prof = interval_profile(dates[valid]) if valid.any() else {}
    missing_ts = 0
    if use and valid.any():
        try:
            missing_ts = len(missing_timestamps(df.assign(**{date_col: dates}), date_col, use))
        except Exception:
            missing_ts = 0
    non_numeric = int((tgt.isna() & df[target_col].notna()).sum())
    miss_target = int(df[target_col].isna().sum())
    s = pd.Series(tgt.values, index=dates)[valid.values].groupby(level=0).mean().sort_index()
    out_n = int(detect_outliers(s.dropna(), thresh=outlier_thresh, season=SEASON_LENGTH.get(use, 1) if use else 1).sum()) if len(s.dropna()) >= 8 else 0
    neg = int((tgt < 0).sum())
    exact_dup = int(df.duplicated().sum())
    pen = {"invalid dates": min(25, 100 * bad / max(n, 1) * 2), "missing/non-numeric target": min(25, 100 * (miss_target + non_numeric) / max(n, 1) * 2),
           "duplicate timestamps": min(10, 100 * dup / max(n, 1)), "missing periods": min(20, 100 * missing_ts / max(len(s) + missing_ts, 1) * 1.5),
           "outliers": min(10, 100 * out_n / max(len(s), 1) * 2), "irregular spacing": min(10, prof.get("irregular_share", 0) * 30)}
    score = float(max(0, 100 - sum(pen.values())))
    issues = []
    if bad: issues.append(f"{bad} unparseable dates")
    if miss_target + non_numeric: issues.append(f"{miss_target + non_numeric} missing/non-numeric target values")
    if dup: issues.append(f"{dup} duplicate timestamps (aggregated)" + (f"; {exact_dup} are exact duplicate rows - consider 'mean' or 'last' instead of 'sum'" if exact_dup else ""))
    if missing_ts: issues.append(f"{missing_ts} missing periods on the {use} grid")
    if out_n: issues.append(f"{out_n} outliers flagged")
    if not prof.get("is_regular", True): issues.append(f"irregular spacing ({prof['irregular_share']:.0%} of gaps off the modal {prof['modal_gap_days']:.0f}-day gap)")
    if neg: issues.append(f"{neg} negative target values")
    return {"rows": n, "invalid_dates": bad, "missing_target": miss_target, "non_numeric_target": non_numeric,
            "duplicate_timestamps": dup, "exact_duplicate_rows": exact_dup, "missing_periods": missing_ts, "outliers": out_n, "negative_values": neg,
            "detected_frequency": det, "intervals": prof, "score": score, "penalties": pen, "issues": issues,
            "date_range": (s.index.min(), s.index.max()) if len(s) else (None, None)}
