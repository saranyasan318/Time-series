
import numpy as np
import pandas as pd

from exploratory_analysis.eda import dominant_cycle, seasonal_profile, seasonality_strength, sudden_changes, trend_summary
from preprocessing.cleaning import detect_outliers

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLS = ["Insight", "Type", "Category", "Recommendation", "Evidence"]


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def comparison_base(hist, fc, freq=None):
    
    from config import YEAR_PERIODS
    hist = hist.dropna()
    yp, h, n = YEAR_PERIODS.get(freq), len(fc), len(hist)
    if yp and n >= yp and h <= yp:
        return hist.iloc[n - yp:n - yp + h].values.astype(float), f"the same {h} periods one year earlier"
    k = min(h, n)
    return hist.iloc[-k:].values.astype(float), f"the last {k} observed periods"


def forecast_growth(hist, fc, freq=None):
   
    b, _ = comparison_base(hist, fc, freq)
    base = float(np.mean(b))
    return (float(fc["forecast"].mean()) / base - 1) * 100 if base else np.nan, base


def forecast_kpis(hist, fc, level=0.95, freq=None):
    
    g, base = forecast_growth(hist.dropna(), fc, freq)
    b, basis = comparison_base(hist, fc, freq)
    prev_total = float(np.sum(b))
    total = float(fc["forecast"].sum())
    ip, tp = fc["forecast"].idxmax(), fc["forecast"].idxmin()
    return {"forecast_total": total, "previous_total": prev_total,
            "total_change_%": (total / prev_total - 1) * 100 if prev_total else np.nan,
            "basis": basis, "total_lower": float(fc["lower"].sum()), "total_upper": float(fc["upper"].sum()),
            "avg_forecast": float(fc["forecast"].mean()), "recent_avg": base, "growth_%": g,
            "peak_date": fc.loc[ip, "date"], "peak_value": float(fc.loc[ip, "forecast"]),
            "low_date": fc.loc[tp, "date"], "low_value": float(fc.loc[tp, "forecast"]),
            "mean_interval_width_%": float(((fc["upper"] - fc["lower"]) / fc["forecast"].abs().replace(0, np.nan)).mean() * 100)}


def _season_label(key, freq):
    k = int(key)
    return {"MS": MONTHS[k - 1] if 1 <= k <= 12 else str(k), "QS": f"Q{k}", "D": DAYS[k % 7], "W": f"week {k}"}[freq]


def generate_insights(hist, fc, holdout_row, freq, selected, level=0.95, threshold=5.0, coverage=None,
                      baseline_gain=None, mase=None):
    """hist: Series; fc: DataFrame(date, forecast, lower, upper); holdout_row: dict of metrics for the selected model."""
    out = []
    hist = hist.dropna()
    kp = forecast_kpis(hist, fc, level, freq)
    g, base, fav, basis = kp["growth_%"], kp["recent_avg"], kp["avg_forecast"], kp["basis"]
    k = min(len(fc), len(hist))
    ev = f"forecast avg {_fmt(fav)} vs baseline avg {_fmt(base)} ({basis})"

    if g > threshold:
        out.append(("Expected growth", "positive", "Outlook", f"Average forecast is {g:.1f}% above {basis}; plan capacity, staffing and stock for higher volume.", ev))
    elif g < -threshold:
        out.append(("Expected decline", "warning", "Risk", f"Average forecast is {abs(g):.1f}% below {basis}; review pricing, promotion and cost plans.", ev))
    else:
        out.append(("Stable outlook", "neutral", "Outlook", f"Forecast is within +/-{threshold:.0f}% of {basis} ({g:+.1f}%); plan for steady demand.", ev))
    out.append(("Forecast total", "info", "Outlook",
                f"Expected total over the horizon is {_fmt(kp['total_lower'])} to {_fmt(kp['total_upper'])} ({level:.0%} interval), "
                f"central estimate {_fmt(kp['forecast_total'])}, {kp['total_change_%']:+.1f}% vs {basis}.",
                f"{_fmt(kp['forecast_total'])} vs {_fmt(kp['previous_total'])}; interval bounds are summed per period (a conservative, wider range)"))

    tr = trend_summary(hist)
    out.append(("Historical trend", "positive" if tr["direction"] == "growth" else "warning" if tr["direction"] == "decline" else "neutral", "History",
                f"Over the full history the series shows {tr['direction']} ({tr['pct_of_mean_per_period']:+.2f}% of the mean per period, linear fit R2={tr['r2']:.2f}).",
                f"slope {tr['slope']:+.2f} per period"))
    if len(hist) >= 2 * max(k, 4) and k >= 2:
        a, b = float(hist.iloc[-k:].sum()), float(hist.iloc[-2 * k:-k].sum())
        if b:
            out.append(("Recent momentum", "positive" if a > b else "warning", "History",
                        f"The latest {k} periods are {(a / b - 1) * 100:+.1f}% versus the {k} periods before them.", f"{_fmt(a)} vs {_fmt(b)}"))

    hist_avg = float(hist.mean())
    pk, lo = fc.loc[fc["forecast"].idxmax()], fc.loc[fc["forecast"].idxmin()]
    out.append(("Peak period", "info", "Opportunity", f"Highest forecast is {pk['date']:%d %b %Y} at {_fmt(pk['forecast'])} ({(pk['forecast'] / hist_avg - 1) * 100:+.0f}% vs the historical average). Secure stock/capacity ahead of it.",
                f"upper bound {_fmt(pk['upper'])} ({level:.0%} interval)"))
    out.append(("Inventory buffer", "info", "Planning", f"For the peak period, a buffer of about {_fmt(pk['upper'] - pk['forecast'])} above the point forecast covers the upper interval bound.",
                f"upper - forecast = {_fmt(pk['upper'])} - {_fmt(pk['forecast'])}"))
    if lo["forecast"] < hist_avg * 0.9:
        out.append(("Soft period", "warning", "Risk", f"Lowest forecast is {lo['date']:%d %b %Y} at {_fmt(lo['forecast'])}, {abs(lo['forecast'] / hist_avg - 1) * 100:.0f}% below the historical average; avoid over-stocking and consider promotions.",
                    f"lowest forecast {_fmt(lo['forecast'])} vs average {_fmt(hist_avg)}"))

    try:
        strength, prof = seasonality_strength(hist, freq), seasonal_profile(hist, freq)
        if strength > 0.2:
            top, bot = prof.idxmax(), prof.idxmin()
            out.append(("Seasonal pattern", "info", "Opportunity", f"Strongest {('month' if freq == 'MS' else 'period')} is {_season_label(top, freq)} ({(prof.max() - 1) * 100:+.0f}% vs average) and weakest is {_season_label(bot, freq)} ({(prof.min() - 1) * 100:+.0f}%); schedule promotions and procurement around this cycle.",
                        f"seasonal range {strength:.2f} x mean"))
        else:
            out.append(("Weak seasonality", "neutral", "History", "No strong seasonal swing detected; demand planning can rely on level and trend.", f"seasonal range {strength:.2f} x mean"))
    except Exception:
        pass

    down = float(fc["lower"].mean())
    if base and down < base * (1 - threshold / 100):
        out.append(("Downside risk", "warning", "Risk", f"In the lower-bound scenario demand averages {_fmt(down)}, {(1 - down / base) * 100:.1f}% below the baseline; a revenue shortfall is plausible.", f"lower bound avg {_fmt(down)} vs baseline avg {_fmt(base)}"))
    up = float(fc["upper"].mean())
    if base and up > base * (1 + threshold / 100):
        out.append(("Upside potential", "positive", "Opportunity", f"In the upper-bound scenario demand averages {_fmt(up)}, {(up / base - 1) * 100:.1f}% above the baseline; keep flexible capacity to capture it.", f"upper bound avg {_fmt(up)} vs baseline avg {_fmt(base)}"))
    width = kp["mean_interval_width_%"]
    if np.isfinite(width) and width > 40:
        out.append(("High uncertainty", "warning", "Risk", f"Prediction interval is wide (about {width:.0f}% of the forecast); use scenario planning and re-forecast frequently.", f"mean width {width:.0f}%"))

    try:
        from config import SEASON_LENGTH
        mask = detect_outliers(hist, season=SEASON_LENGTH.get(freq, 1))
        sc = sudden_changes(hist)
        if mask.any() or len(sc):
            dates = ", ".join(f"{d:%b %Y}" for d in list(hist.index[mask])[:4])
            out.append(("Anomalies in history", "warning", "Risk", f"{int(mask.sum())} unusual observations and {len(sc)} abrupt period-to-period changes were found" + (f" (e.g. {dates})" if dates else "") + "; verify whether they were one-off events before relying on the trend.",
                        f"robust z-score > 3.5; |z| > 3 on period change"))
    except Exception:
        pass

    mape = holdout_row.get("MAPE", np.nan)
    if np.isfinite(mape):
        lvl = "high" if mape < 10 else "moderate" if mape < 20 else "low"
        extra = []
        if baseline_gain is not None and np.isfinite(baseline_gain):
            extra.append(f"{baseline_gain:+.0f}% RMSE vs best baseline")
        if coverage is not None:
            extra.append(f"hold-out interval coverage {coverage:.0%} (nominal {level:.0%})")
        if mase is not None and np.isfinite(mase):
            extra.append(f"MASE {mase:.2f}")
        out.append(("Forecast reliability", "info" if lvl != "low" else "warning", "Confidence",
                    f"{selected} had hold-out MAPE {mape:.1f}% -> {lvl} reliability. Forecasts are estimates, not guarantees.",
                    "; ".join([f"hold-out MAPE {mape:.1f}%, sMAPE {holdout_row.get('sMAPE', np.nan):.1f}%"] + extra)))
    if coverage is not None and coverage < level - 0.25:
        out.append(("Intervals may be too narrow", "warning", "Confidence", f"Only {coverage:.0%} of hold-out actuals fell inside the {level:.0%} interval; treat the bands as optimistic.", f"coverage {coverage:.0%}"))
    if baseline_gain is not None and np.isfinite(baseline_gain) and baseline_gain <= 0:
        out.append(("No gain over baseline", "warning", "Confidence", "The selected model did not beat the best naive baseline on the hold-out; the simpler baseline is a defensible alternative.", f"RMSE vs baseline {baseline_gain:+.1f}%"))
    return pd.DataFrame(out, columns=COLS)


def limitations(n_obs, season, selected):
    items = [f"Based on {n_obs} observations; short histories give unstable seasonal estimates.",
             "Assumes patterns of the past continue; promotions, price changes or shocks are not modelled.",
             "Prediction intervals are model-based approximations and may under-cover."]
    if n_obs < 3 * season: items.append("Fewer than three seasonal cycles are available, so seasonality is weakly identified.")
    if selected in ("Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost"): items.append("ML models forecast recursively, so errors can accumulate on long horizons.")
    return items
