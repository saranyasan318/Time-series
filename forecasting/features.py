"""Leak-free lag, rolling and calendar features for ML forecasting.
Every feature for time t uses only values strictly before t; calendar features use the known date."""
import numpy as np
import pandas as pd

from config import LAGS_BY_FREQ


def choose_lags(freq, n):
    lags = [l for l in LAGS_BY_FREQ[freq] if l <= max(2, n // 3)]
    return lags or [1]


def calendar_features(date):
    d = pd.Timestamp(date)
    return [d.month, d.quarter, d.dayofweek, d.day, int(d.isocalendar()[1])]


def row_features(history, date, lags, windows):
    """Features for the next point after `history` (1-D array)."""
    h = np.asarray(history, float)
    f = [h[-l] for l in lags]
    for w in windows:
        seg = h[-w:]
        f += [seg.mean(), seg.std()]
    return f + calendar_features(date)


def build_supervised(y, dates, lags, windows):
    """Training matrix X, target t. Row i uses y[:i] only."""
    start = max(lags + windows)
    X, t = [], []
    for i in range(start, len(y)):
        X.append(row_features(y[:i], dates[i], lags, windows))
        t.append(y[i])
    return np.array(X), np.array(t)
