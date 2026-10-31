"""Model selection and final evaluation.

Procedure (prevents selection bias):
 1. Hold out the LAST `test_h` observations. They are never used for model choice.
 2. Rank models by chronological cross-validation (expanding, rolling or walk-forward) on the remaining data.
    Pre-processing is re-fitted inside every training window.
 3. Select the model with the best pooled CV score on the primary metric (default RMSE). Training error is never used.
 4. Refit every model on the full train/validation data and score the hold-out ONCE. The hold-out score of the
    selected model is the unbiased estimate; other models' hold-out scores are informational only.
Models that fail are recorded in `errors` and excluded; they never abort the run.
"""
import time

import numpy as np
import pandas as pd

from config import DEFAULT_PRIMARY_METRIC, WALK_FORWARD_STEPS
from evaluation.metrics import all_metrics
from forecasting.models import BASELINES
from forecasting.pipeline import run_model
from config import OUTLIER_METHODS
from preprocessing.cleaning import PrepConfig, apply_prep, detect_outliers
from validation.splits import assert_no_leakage, chronological_split, time_series_folds

METRIC_COLS = ["MAE", "MSE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"]


def evaluate_models(series, freq, season, cfg: PrepConfig, model_names, horizon, n_splits=3,
                    mode="expanding", level=0.95, primary_metric=DEFAULT_PRIMARY_METRIC, progress=None):
    n = len(series)
    test_h = int(min(horizon, max(2, n // 4)))
    trainval, holdout = chronological_split(series, test_h)
    assert_no_leakage(trainval, holdout)
    # large enough for seasonal models (2 cycles + 2) whenever the data allows it, so they are not unfairly excluded
    min_train = max(12, len(trainval) // 2, min(2 * season + 2, int(0.8 * len(trainval))))
    if mode == "walk-forward":
        n_splits = min(max(n_splits, WALK_FORWARD_STEPS), max(len(trainval) - min_train, 1))
        cv_h = 1
    else:
        cv_h = max(1, min(test_h, (len(trainval) - min_train) // max(n_splits, 1)))
    folds = time_series_folds(len(trainval), cv_h, n_splits, min_train, mode, window=min_train)
    if not folds:
        raise ValueError("Not enough history for cross-validation; reduce folds or horizon.")
    notes = []
    if test_h < horizon:
        notes.append(f"Hold-out shortened to {test_h} periods (selected horizon {horizon} is too long for this history).")

    # When an outlier treatment is chosen, flagged data errors are excluded from SCORING (never from the ground truth
    # silently): models are trained on treated data and judged on the observations that are not flagged outliers.
    score_mask = pd.Series(False, index=series.index)
    if cfg.outlier_method != OUTLIER_METHODS[0]:
        filled = series.interpolate(limit_area="inside")
        score_mask = detect_outliers(filled.dropna(), thresh=cfg.outlier_threshold, season=season if season > 1 else 1)
        score_mask = score_mask.reindex(series.index, fill_value=False) & series.notna()
        if score_mask.any():
            notes.append(f"{int(score_mask.sum())} flagged outlier(s) were excluded from error scoring (outlier treatment is on): "
                         + ", ".join(f"{d:%Y-%m-%d}" for d in series.index[score_mask][:5]) + ".")

    def scored(w):   # actuals with flagged outliers masked out
        v = w.values.astype(float).copy()
        v[score_mask.reindex(w.index, fill_value=False).values] = np.nan
        return v

    # MASE scale: in-sample (seasonal) naive error of the train/validation data only
    scale_m = season if season > 1 and len(trainval.dropna()) > 2 * season else 1
    scale_train = apply_prep(trainval, cfg, season)[0].values

    cv_rows, errors, fold_rows, timing = [], {}, [], {}
    names = list(model_names)
    for k, name in enumerate(names):
        if progress:
            progress(k / max(len(names), 1), f"Cross-validating {name}")
        acts, preds, fold_rmse = [], [], []
        t0 = time.perf_counter()
        try:
            for fi, (s0, e, te) in enumerate(folds):
                train, _ = apply_prep(trainval.iloc[s0:e], cfg, season)   # preprocessing fitted on training only
                test = trainval.iloc[e:te]
                assert_no_leakage(train, test)
                p = run_model(name, train, len(test), season, freq, level)["mean"]
                acts += list(scored(test)); preds += list(p)
                fm = all_metrics(scored(test), p, train.values, scale_m)
                fold_rows.append({"Model": name, "Fold": fi + 1, "Train end": train.index[-1], "Test start": test.index[0],
                                  "Test end": test.index[-1], **fm})
                fold_rmse.append(fm["RMSE"])
        except Exception as ex:
            errors[name] = f"{type(ex).__name__}: {ex}"
            continue
        timing[name] = time.perf_counter() - t0
        m = all_metrics(acts, preds, scale_train, scale_m)
        cv_rows.append({"Model": name, **m, "CV folds": len(folds), "Fold RMSE std": float(np.nanstd(fold_rmse)),
                        "Fit time (s)": timing[name]})
    if not cv_rows:
        raise ValueError("No model could be evaluated: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    pm = primary_metric if primary_metric in METRIC_COLS else DEFAULT_PRIMARY_METRIC
    cv = pd.DataFrame(cv_rows)
    cv = cv.assign(_k=cv[pm].fillna(np.inf)).sort_values("_k").drop(columns="_k").reset_index(drop=True)
    cv.insert(0, "Rank", np.arange(1, len(cv) + 1))
    selected = cv.iloc[0]["Model"]

    train_full, _ = apply_prep(trainval, cfg, season)
    rows, pred_df, bands, meta = [], pd.DataFrame({"date": holdout.index, "Actual": holdout.values}), {}, {}
    for name in cv["Model"]:
        try:
            res = run_model(name, train_full, test_h, season, freq, level)
        except Exception as ex:   # a model that CV-ed fine but fails on the full window is reported, not fatal
            errors[name] = f"hold-out refit failed - {type(ex).__name__}: {ex}"
            continue
        pred_df[name] = res["mean"]; bands[name] = (res["lower"], res["upper"]); meta[name] = res["meta"]
        rows.append({"Model": name, **all_metrics(scored(holdout), res["mean"], train_full.values, scale_m)})
    if not rows:
        raise ValueError("No model could be refit for the hold-out evaluation.")
    hold = pd.DataFrame(rows)
    base = hold[hold["Model"].isin(BASELINES)]["RMSE"]
    if len(base):
        hold["RMSE vs best baseline (%)"] = (1 - hold["RMSE"] / base.min()) * 100
    hold["Selected (by CV)"] = hold["Model"] == selected
    hold = hold.sort_values("RMSE").reset_index(drop=True)
    residuals = pd.DataFrame({"date": holdout.index})
    for name in bands:
        residuals[name] = holdout.values - pred_df[name].values
    # empirical coverage of the prediction interval on the hold-out (honest check of the stated level)
    coverage = {nm: float(np.mean((holdout.values >= lo) & (holdout.values <= hi))) for nm, (lo, hi) in bands.items()}
    return {"cv": cv, "holdout": hold, "selected": selected, "predictions": pred_df, "bands": bands,
            "meta": meta, "errors": errors, "notes": notes, "test_h": test_h, "cv_h": cv_h, "level": level,
            "primary_metric": pm, "residuals": residuals, "coverage": coverage, "fold_details": pd.DataFrame(fold_rows),
            "train_period": (trainval.index.min(), trainval.index.max()),
            "test_period": (holdout.index.min(), holdout.index.max()),
            "cv_folds": [(trainval.index[s], trainval.index[e - 1], trainval.index[te - 1]) for s, e, te in folds],
            "cv_mode": mode, "mase_period": scale_m}
