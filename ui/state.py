
import io

import numpy as np
import pandas as pd
import streamlit as st

from config import FREQUENCIES, SEASON_LENGTH
from exploratory_analysis import eda
from forecasting.pipeline import future_forecast
from preprocessing.cleaning import (PrepConfig, apply_prep, detect_frequency, guess_date_col, guess_target_col,
                                    prepare_series, validate_dataset)

S = st.session_state

PAGES = ["Overview Dashboard", "Dataset Upload", "Data Validation & Quality", "Exploratory Data Analysis", "Data Preparation",
         "Forecasting Lab", "Model Evaluation & Comparison", "Prediction Dashboard", "Business Intelligence & Insights",
         "Reports & Downloads", "Q3 – Time Series Decision Challenge"]

DEFAULTS = {"nav": PAGES[0], "raw": None, "dataset_name": None, "ds": None, "series": None, "build_log": [], "sig": None,
            "freq": "MS", "cfg": PrepConfig(), "ev": None, "fcs": {}, "meta": {}, "horizon": None, "level": 0.95, "up_id": None}


def init():
    for k, v in DEFAULTS.items():
        S.setdefault(k, v)


def goto(page):
    S["nav"] = page


def reset_results():
    S.update(ev=None, fcs={}, meta={}, horizon=None)


def load_dataset(df, name):
    """Install a new raw dataset (never modified afterwards) and auto-configure sensible defaults."""
    S.update(raw=df, dataset_name=name, series=None, sig=None, build_log=[])
    reset_results()
    S["cfg"] = PrepConfig()
    S["ds"] = auto_configure(df)
    build_series()


def auto_configure(df):
    dc = guess_date_col(df)
    tc = guess_target_col(df, dc)
    dates = pd.to_datetime(df[dc], errors="coerce", format="mixed") if not pd.api.types.is_numeric_dtype(df[dc]) else pd.to_datetime(df[dc], errors="coerce")
    det = detect_frequency(dates.dropna()) if dates.notna().any() else None
    return {"date_col": dc, "target_col": tc, "group_col": None, "group_value": None, "freq": det or "MS",
            "agg": "sum", "date_format": None, "detected": det}


def build_series():
   
    raw, ds = S["raw"], S["ds"]
    if raw is None or ds is None or ds["target_col"] is None:
        S["series"] = None
        return None
    sig = (id(raw), len(raw), tuple(sorted((k, str(v)) for k, v in ds.items() if k != "detected")))
    if S["series"] is not None and S["sig"] == sig:
        return S["series"]
    rep = validate_dataset(raw, ds["date_col"], ds["target_col"], ds["group_col"], ds["freq"], date_format=ds["date_format"])
    if rep["errors"]:
        S.update(series=None, sig=None, build_log=[])
        return None
    s, log = prepare_series(raw, ds["date_col"], ds["target_col"], ds["freq"], ds["group_col"], ds["group_value"], ds["agg"], ds["date_format"])
    S.update(series=s, sig=sig, build_log=log, freq=ds["freq"])
    reset_results()
    return s


def season():
    return SEASON_LENGTH[S["freq"]]


def ready():
    return S["series"] is not None


def prepped():
    return apply_prep(S["series"], S["cfg"], season())


@st.cache_data(show_spinner=False)
def read_csv_cached(data: bytes):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def cached_characterise(series, freq, season_len):
    return eda.characterise(series, freq, season_len)


@st.cache_data(show_spinner=False)
def cached_decompose(series, season_len, model):
    return eda.decompose(series, season_len, model)


@st.cache_data(show_spinner=False)
def cached_acf(values, nlags):
    return eda.acf_values(values, nlags), eda.pacf_values(values, nlags)


@st.cache_data(show_spinner=False)
def cached_future(series, freq, season_len, cfg, model, horizon, level):
    return future_forecast(series, freq, season_len, cfg, model, horizon, level)
