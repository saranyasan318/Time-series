"""Synthetic but realistic datasets: the 48-month sample and the eight verification scenarios."""
import numpy as np
import pandas as pd

_SEASON = np.array([-0.12, -0.10, -0.04, 0.00, 0.03, 0.05, 0.02, 0.04, 0.00, 0.06, 0.15, 0.22])


def make_sample_sales(months: int = 48, seed: int = 42) -> pd.DataFrame:
    """Monthly sales for three categories: trend + Nov/Dec seasonal peak + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=months, freq="MS")
    t = np.arange(months)
    rows = []
    for cat, base, slope in [("Electronics", 120000, 1900), ("Apparel", 80000, 1100), ("Home", 60000, 700)]:
        sales = (base + slope * t) * (1 + _SEASON[idx.month - 1] + rng.normal(0, 0.025, months))
        units = sales / rng.uniform(48, 52, months)
        rows.append(pd.DataFrame({"Date": idx, "Category": cat, "Sales": sales.round(2), "Units": units.round(0)}))
    return pd.concat(rows, ignore_index=True).sort_values(["Date", "Category"]).reset_index(drop=True)


def _base(n=48, seed=1, trend=1500.0, seasonal=True, noise=0.03, freq="MS"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq)
    y = (50000 + trend * np.arange(n)) * (1 + (_SEASON[idx.month - 1] if seasonal else 0) + rng.normal(0, noise, n))
    return pd.DataFrame({"Date": idx, "Sales": y})


def make_test_scenarios() -> dict:
    """The eight datasets required by the verification plan."""
    clean = _base()
    missing = clean.copy(); missing.loc[[5, 6, 20, 33], "Sales"] = np.nan
    dup = pd.concat([clean, clean.iloc[[3, 10, 11]]], ignore_index=True).sort_values("Date").reset_index(drop=True)
    outl = clean.copy(); outl.loc[15, "Sales"] *= 4.0; outl.loc[30, "Sales"] *= 0.1
    seasonal = _base(trend=0.0, seed=2, noise=0.02)
    up = _base(trend=2500.0, seasonal=False, seed=3)
    down = _base(trend=-600.0, seasonal=False, seed=4)
    irregular = clean.drop(index=[4, 5, 6, 17, 18, 29, 30, 31, 40]).reset_index(drop=True)
    short = clean.iloc[:10].copy()
    return {"clean": clean, "missing": missing, "duplicates": dup, "outliers": outl, "seasonal": seasonal,
            "upward_trend": up, "downward_trend": down, "irregular": irregular, "insufficient": short}


def make_frequency_datasets(seed: int = 7) -> dict:
    """Datasets at other frequencies / lengths for verification: daily (weekly + yearly seasonality), weekly,
    quarterly, long monthly, short monthly and day-first date strings."""
    rng = np.random.default_rng(seed)
    out = {}
    d = pd.date_range("2022-01-01", periods=730, freq="D")
    t = np.arange(730)
    y = 200 + 0.15 * t + 25 * np.sin(2 * np.pi * t / 365.25) + 30 * (d.dayofweek >= 5) + rng.normal(0, 6, 730)
    out["daily"] = pd.DataFrame({"Date": d, "Sales": y.round(2)})
    w = pd.date_range("2021-01-03", periods=156, freq="W")
    tw = np.arange(156)
    out["weekly"] = pd.DataFrame({"Date": w, "Sales": (1000 + 4 * tw + 120 * np.sin(2 * np.pi * tw / 52.18) + rng.normal(0, 25, 156)).round(1)})
    q = pd.date_range("2016-01-01", periods=32, freq="QS")
    tq = np.arange(32)
    out["quarterly"] = pd.DataFrame({"Date": q, "Sales": (5000 + 90 * tq + np.tile([-300, 100, -50, 400], 8) + rng.normal(0, 60, 32)).round(1)})
    out["long_monthly"] = _base(n=120, seed=11, trend=900.0)
    out["short_monthly"] = _base(n=26, seed=12)
    dfm = _base(n=36, seed=13)
    out["dayfirst_strings"] = pd.DataFrame({"Date": dfm["Date"].dt.strftime("%d/%m/%Y"), "Sales": dfm["Sales"].round(2)})
    return out
