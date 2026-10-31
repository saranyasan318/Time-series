
from __future__ import annotations

import math
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SALES = [120, 130, 125, 140, 152, 149, 165, 170, 190, 185, 205, 215]
FIRST_TEST_WEEK = 5          
MA_WINDOW = 3               
METHODS = ("Naive last value", "Trailing 3-week average")



def validate_values(values: Iterable, minimum: int = MA_WINDOW) -> np.ndarray:
    try:
        y = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Values must all be numeric.") from exc
    if y.ndim != 1 or len(y) < minimum:
        raise ValueError(f"Provide at least {minimum} weekly observations.")
    if not np.isfinite(y).all():
        raise ValueError("Values must not contain blanks, NaN or infinity.")
    if (y < 0).any():
        raise ValueError("Units sold cannot be negative.")
    return y

def naive_forecast(history: np.ndarray) -> float:
    return float(history[-1])


def trailing_mean_forecast(history: np.ndarray, window: int = MA_WINDOW) -> float:
    if len(history) < window:
        raise ValueError(f"Need at least {window} observations for a trailing mean.")
    return float(np.mean(history[-window:]))


def forecast_next(values: Iterable) -> dict:
    """Forecast for the week after the last observation, with both methods."""
    y = validate_values(values, MA_WINDOW)
    return {
        "target_week": len(y) + 1,
        METHODS[0]: naive_forecast(y),
        METHODS[1]: trailing_mean_forecast(y),
    }



def expanding_backtest(values: Iterable, first_test_week: int = FIRST_TEST_WEEK) -> pd.DataFrame:
    """Expanding-window one-step-ahead backtest.

    For each target week t >= first_test_week the training window is weeks 1..t-1
    (it grows by one each step) and both methods forecast week t. Nothing at or after
    week t is ever visible to the forecast.
    """
    if first_test_week <= MA_WINDOW:
        raise ValueError("first_test_week must exceed the trailing-mean window so every forecast is defined.")
    y = validate_values(values, first_test_week)
    rows = []
    for t in range(first_test_week, len(y) + 1):          
        history = y[: t - 1]
        actual = float(y[t - 1])
        f_naive = naive_forecast(history)
        f_avg = trailing_mean_forecast(history)
        rows.append({
            "Week": t,
            "Training weeks": len(history),
            "Actual": actual,
            "Naive forecast": f_naive,
            "Naive error": actual - f_naive,                
            "Naive absolute error": abs(actual - f_naive),
            "3-week average forecast": f_avg,
            "3-week average error": actual - f_avg,
            "3-week average absolute error": abs(actual - f_avg),
        })
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> dict:
   
    n_abs, a_abs = bt["Naive absolute error"], bt["3-week average absolute error"]
    return {
        "n_weeks": int(len(bt)),
        "naive_mae": float(n_abs.mean()),
        "avg_mae": float(a_abs.mean()),
        "naive_bias": float(bt["Naive error"].mean()),
        "avg_bias": float(bt["3-week average error"].mean()),
        "naive_wins": int((n_abs < a_abs).sum()),
        "avg_wins": int((a_abs < n_abs).sum()),
        "ties": int((n_abs == a_abs).sum()),
    }


def select_baseline(naive_mae: float, avg_mae: float) -> str:
    
    return METHODS[0] if naive_mae <= avg_mae else METHODS[1]


def leaky_neighbour_mae(values: Iterable, first_test_week: int = FIRST_TEST_WEEK) -> float:
 
    y = validate_values(values, first_test_week + 1)
    errs = [abs(y[t - 1] - (y[t - 2] + y[t]) / 2) for t in range(first_test_week, len(y))]
    return float(np.mean(errs))


# ------------------------------------------------------------------------ (c) dates
def audit_weekly_dates(dates=None, *, n_obs: Optional[int] = None, assumed_start: Optional[str] = None,
                       freq: str = "W-SUN") -> dict:
 
    if freq not in {"W-SUN", "W-MON"}:
        raise ValueError("freq must be 'W-SUN' or 'W-MON'.")

    if dates is None:
        if n_obs is None or n_obs < 1:
            raise ValueError("Without dates, n_obs (>= 1) is required.")
        idx = None
        if assumed_start is not None:
            idx = pd.date_range(assumed_start, periods=n_obs, freq=freq)
        return {"mode": "assumed (no timestamps supplied)", "n_input": int(n_obs), "invalid": 0,
                "exact_duplicates": [], "same_week_duplicates": [], "missing_weeks": [],
                "out_of_order": False, "off_cadence": 0, "ok": None,
                "assumed_index": idx,
                "note": "No timestamps: gaps/duplicates cannot be observed. Consecutive weekly spacing is an "
                        "ASSUMPTION, so zero issues here is by construction, not evidence."}

    with warnings.catch_warnings():      # mixed/unknown formats are expected here; unparseable -> counted invalid
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    invalid = int(parsed.isna().sum())
    valid = parsed.dropna().reset_index(drop=True)
    empty = {"mode": "timestamps", "n_input": int(len(parsed)), "invalid": invalid, "exact_duplicates": [],
             "same_week_duplicates": [], "missing_weeks": [], "out_of_order": False, "off_cadence": 0,
             "ok": invalid == 0 and len(parsed) > 0, "note": ""}
    if valid.empty:
        empty["ok"] = False
        empty["note"] = "No valid dates to audit."
        return empty

    day = valid.dt.normalize()
    exact = sorted(day[day.duplicated(keep=False)].unique())
    periods = valid.dt.to_period(freq)
    same_week = sorted(periods[periods.duplicated(keep=False)].unique())
    # a pair of identical dates is also a same-week pair; report the extra "different-day" cases separately
    same_week_only = [p for p in same_week if len(set(day[periods == p])) > 1]
    unique_p = periods.drop_duplicates().sort_values()
    expected = pd.period_range(unique_p.iloc[0], unique_p.iloc[-1], freq=freq)
    missing = [p.end_time.normalize() for p in expected.difference(pd.PeriodIndex(unique_p))]
    out_of_order = bool((valid.diff().dropna() < pd.Timedelta(0)).any())
    off_cadence = int(((day.dt.dayofweek) != (6 if freq == "W-SUN" else 0)).sum())

    empty.update({
        "exact_duplicates": [pd.Timestamp(d) for d in exact],
        "same_week_duplicates": [p.end_time.normalize() for p in same_week_only],
        "missing_weeks": missing, "out_of_order": out_of_order, "off_cadence": off_cadence,
    })
    empty["ok"] = not (invalid or exact or same_week_only or missing or out_of_order)
    empty["note"] = ("Clean under the selected convention." if empty["ok"]
                     else "Resolve issues before treating this as a regular weekly series.")
    return empty


def validate_weekly_dates(dates, freq="W-SUN"):

    dates = list(dates)
    if not dates:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([]), 0
    r = audit_weekly_dates(dates, freq=freq)
    dup = sorted({pd.Timestamp(d).to_period(freq).end_time.normalize() for d in r["exact_duplicates"]}
                 | set(r["same_week_duplicates"]))
    return pd.DatetimeIndex(dup), pd.DatetimeIndex(r["missing_weeks"]), r["invalid"]


# ----------------------------------------------------------------------- (d) alert rule
def alert_tolerance(abs_errors: Iterable, coverage: float = 0.80) -> dict:

    if not 0.5 <= coverage < 1:
        raise ValueError("coverage must be in [0.5, 1).")
    e = np.sort(np.asarray(list(abs_errors), dtype=float))
    e = e[np.isfinite(e)]
    if len(e) == 0:
        raise ValueError("Backtest errors are required to calculate an alert tolerance.")
    n = len(e)
    k = math.ceil((n + 1) * coverage)
    ok = k <= n
    return {"half_width": float(e[min(k, n) - 1]), "n_errors": n, "rank_used": min(k, n),
            "coverage_target": coverage, "guarantee_available": ok,
            "min_errors_needed": math.ceil(coverage / (1 - coverage))}


def demand_alert(forecast: float, actual: float, abs_errors: Iterable, coverage: float = 0.80) -> dict:

    if not np.isfinite([forecast, actual]).all():
        raise ValueError("Forecast and actual demand must be finite numbers.")
    if forecast < 0 or actual < 0:
        raise ValueError("Demand values cannot be negative.")
    tol = alert_tolerance(abs_errors, coverage)
    lo, hi = max(0.0, forecast - tol["half_width"]), forecast + tol["half_width"]
    direction = "above" if actual > hi else "below" if actual < lo else "within"
    return {**tol, "forecast": float(forecast), "actual": float(actual), "lower": lo, "upper": hi,
            "absolute_error": abs(actual - forecast), "alert": direction != "within", "direction": direction}


def decision_memo(values: Iterable = SALES, coverage: float = 0.80) -> str:
    
    y = validate_values(values, FIRST_TEST_WEEK)
    fc = forecast_next(y)
    bt = expanding_backtest(y)
    s = backtest_summary(bt)
    chosen = select_baseline(s["naive_mae"], s["avg_mae"])
    errs_col = "Naive absolute error" if chosen == METHODS[0] else "3-week average absolute error"
    chosen_fc = fc[chosen]
    tol = alert_tolerance(bt[errs_col], coverage)
    leak = leaky_neighbour_mae(y)
    n = s["n_weeks"]
    guarantee = ("reaches the minimum sample for a finite-sample coverage statement, but only if future errors resemble past ones (doubtful on a trending series)"
                 if tol["guarantee_available"] else
                 f"is BELOW the {tol['min_errors_needed']} errors needed for a finite-sample "
                 f"{coverage:.0%} coverage statement, so no coverage is claimed")
    return f"""# Q3 decision memo: weekly demand baseline

## Decision
Adopt **{chosen}** as the working baseline for week {fc['target_week']}: forecast **{chosen_fc:.1f} units**.
It had the lower one-step-ahead MAE over the {n} backtest weeks (5-12). This is a choice between two
simple baselines on one short series. It is **not** evidence of how accurate the forecast will be.

## Evidence
| Method | Week {fc['target_week']} forecast | Backtest MAE | Mean signed error | Weeks it was closer |
|---|---|---|---|---|
| Naive last value | {fc[METHODS[0]]:.1f} | {s['naive_mae']:.3f} | {s['naive_bias']:+.2f} | {s['naive_wins']} of {n} |
| Trailing 3-week average | {fc[METHODS[1]]:.1f} | {s['avg_mae']:.3f} | {s['avg_bias']:+.2f} | {s['avg_wins']} of {n} |

Signed error = actual - forecast. Both mean signed errors are positive: demand kept rising, so both methods under-forecast on average.
The 3-week average lags more because it averages older, lower weeks. That lag is the likely reason it loses,
and it would reverse on a flat or mean-reverting series.

## Validation design
Expanding window, one step ahead: to score week t the methods see weeks 1..t-1 only.
Shuffling observations would let weeks after t inform the forecast of week t. As an illustration, "predicting"
each week from its two neighbours (one of which is in the future) gives MAE {leak:.2f}, lower than the honest
naive forecast here ({s["naive_mae"]:.2f}), even though it uses information no real forecast could have. Shuffling also breaks the trend and autocorrelation the methods depend on.

## Demand alert
Alert if actual demand falls outside forecast +/- **{tol['half_width']:.1f} units** (the
rank-{tol['rank_used']} of {tol['n_errors']} backtest absolute errors of the chosen method, target {coverage:.0%}).
For week {fc['target_week']}: {max(0, chosen_fc - tol['half_width']):.1f} to {chosen_fc + tol['half_width']:.1f} units.
With {tol['n_errors']} errors this {guarantee}.

## Limitations
- {n} errors from one series: MAE, the band and the baseline ranking could all change with a few more weeks.
  No significance test is meaningful at this size.
- The band is symmetric, but errors here are one-sided (bias {s['naive_bias']:+.1f} for naive). Under a continuing trend
  a symmetric band will flag rises late and miss that the forecast is systematically low.
- The rule assumes future errors look like past errors (exchangeability). Trend changes, promotions,
  stock-outs (sales are not demand) and holidays break this.
- Week numbers are assumed to be consecutive weeks. No timestamps were supplied, so missing or
  duplicate calendar dates cannot be detected on this data; the detector is tested on separate dated inputs.
- Recommendation: use the baseline now, record each new week's error, and re-run the comparison (and test
  a trend-aware challenger such as drift or Holt) once there are enough weeks to support it.
"""
