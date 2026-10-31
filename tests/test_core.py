import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MISSING_METHODS, OUTLIER_METHODS
from evaluation.metrics import all_metrics, mae, mape, rmse, smape
from evaluation.selection import evaluate_models
from forecasting.features import build_supervised, row_features
from forecasting.pipeline import future_forecast, run_model
from preprocessing.cleaning import (PrepConfig, apply_prep, detect_frequency, detect_outliers, prepare_series,
                                    treat_missing, validate_dataset)
from utils.sample_data import make_sample_sales, make_test_scenarios
from validation.splits import assert_no_leakage, chronological_split, time_series_folds

HAS_SM = importlib.util.find_spec("statsmodels") is not None
SC = make_test_scenarios()


def series_of(df):
    s, _ = prepare_series(df, "Date", "Sales", "MS")
    return s


class TestMetrics(unittest.TestCase):
    def test_values(self):
        y, p = [100, 200, 300], [110, 190, 330]
        self.assertAlmostEqual(mae(y, p), 16.6667, 3)
        self.assertAlmostEqual(rmse(y, p), np.sqrt((100 + 100 + 900) / 3), 6)
        self.assertAlmostEqual(mape(y, p), np.mean([.1, .05, .1]) * 100, 6)

    def test_mape_zero_safe(self):
        self.assertAlmostEqual(mape([0, 100], [5, 90]), 10.0)
        self.assertTrue(np.isnan(mape([0, 0], [1, 1])))

    def test_smape_bounds(self):
        self.assertLessEqual(smape([1, 2, 3], [100, 200, 300]), 200)
        self.assertEqual(smape([0], [0]) != smape([0], [0]), True)  # NaN

    def test_nan_ignored(self):
        self.assertEqual(mae([1, np.nan, 3], [2, 5, 3]), 0.5)
        self.assertEqual(set(all_metrics([1, 2], [1, 2])), {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"})


class TestSplits(unittest.TestCase):
    def test_chronological(self):
        s = series_of(SC["clean"]); tr, te = chronological_split(s, 6)
        self.assertEqual(len(te), 6); self.assertTrue(assert_no_leakage(tr, te))
        self.assertEqual(te.index[-1], s.index[-1])

    def test_leakage_detected(self):
        s = series_of(SC["clean"])
        with self.assertRaises(AssertionError): assert_no_leakage(s.iloc[:30], s.iloc[20:])

    def test_folds_expanding_and_order(self):
        f = time_series_folds(40, 4, 3, 12)
        self.assertEqual(len(f), 3); self.assertEqual(f[-1][2], 40)
        for s0, e, te in f: self.assertTrue(s0 == 0 and e < te)
        self.assertTrue(all(f[i][1] < f[i + 1][1] for i in range(2)))

    def test_folds_rolling_window_fixed(self):
        f = time_series_folds(40, 4, 3, 12, mode="rolling", window=12)
        self.assertTrue(all(e - s0 == 12 for s0, e, _ in f))


class TestPreprocessing(unittest.TestCase):
    def test_frequency_detection(self):
        self.assertEqual(detect_frequency(SC["clean"]["Date"]), "MS")
        self.assertEqual(detect_frequency(pd.date_range("2024-01-01", periods=30)), "D")
        self.assertEqual(detect_frequency(pd.date_range("2024-01-07", periods=30, freq="W")), "W")

    def test_clean_valid(self):
        r = validate_dataset(SC["clean"], "Date", "Sales")
        self.assertEqual(r["errors"], [])

    def test_insufficient_blocked(self):
        r = validate_dataset(SC["insufficient"], "Date", "Sales", freq="MS")
        self.assertTrue(any("Insufficient" in e for e in r["errors"]))

    def test_duplicates_warned_and_aggregated(self):
        r = validate_dataset(SC["duplicates"], "Date", "Sales")
        self.assertEqual(r["info"]["duplicate_timestamps"], 3)
        s = series_of(SC["duplicates"]); self.assertTrue(s.index.is_unique); self.assertEqual(len(s), 48)

    def test_invalid_dates_and_types(self):
        d = SC["clean"].copy(); d["Date"] = d["Date"].astype(str); d.loc[2, "Date"] = "not a date"; d["Sales"] = d["Sales"].astype(object); d.loc[3, "Sales"] = "abc"
        r = validate_dataset(d, "Date", "Sales")
        self.assertEqual(r["info"]["invalid_dates"], 1); self.assertEqual(r["info"]["non_numeric_target"], 1)

    def test_all_text_target_error(self):
        d = SC["clean"].copy(); d["Sales"] = "x"
        self.assertTrue(validate_dataset(d, "Date", "Sales")["errors"])

    def test_missing_periods_stay_nan_and_raw_preserved(self):
        raw = SC["irregular"].copy(); snapshot = raw.copy()
        s = series_of(raw)
        self.assertEqual(len(s), 48); self.assertGreater(s.isna().sum(), 0)
        pd.testing.assert_frame_equal(raw, snapshot)

    def test_interpolation_only_short_gaps(self):
        s = pd.Series([1., np.nan, 3., np.nan, np.nan, np.nan, 7., 8.], index=pd.date_range("2024-01-01", periods=8, freq="MS"))
        out, n = treat_missing(s, MISSING_METHODS[1])
        self.assertEqual(out.iloc[1], 2.0); self.assertTrue(out.iloc[3:6].isna().all()); self.assertEqual(n, 1)

    def test_seasonal_fill(self):
        s = series_of(SC["missing"]); out, n = treat_missing(s, MISSING_METHODS[2], 12)
        self.assertEqual(out.isna().sum(), 0); self.assertEqual(n, 4)

    def test_outlier_detection_and_treatment(self):
        s = series_of(SC["outliers"]); m = detect_outliers(s, 12)
        self.assertTrue(m.iloc[15] and m.iloc[30])
        out, _ = apply_prep(s, PrepConfig(outlier_method=OUTLIER_METHODS[2]), 12)
        self.assertLess(out.iloc[15], s.iloc[15])

    def test_prep_uses_only_given_window(self):
        s = series_of(SC["missing"]); a, _ = apply_prep(s.iloc[:30], PrepConfig(), 12)
        b, _ = apply_prep(s, PrepConfig(), 12)
        pd.testing.assert_series_equal(a.iloc[:19], b.iloc[:19])  # early values unaffected by future data


class TestFeatures(unittest.TestCase):
    def test_no_leakage_in_features(self):
        y = np.arange(40, dtype=float); d = pd.date_range("2021-01-01", periods=40, freq="MS")
        X, t = build_supervised(y, d, [1, 2, 3], [3])
        # row for target y[i]: lag1 must equal y[i-1]
        self.assertTrue(np.all(X[:, 0] == t - 1)); self.assertTrue(np.all(X[:, 2] == t - 3))
        self.assertTrue(np.all(X[:, 3] == t - 2))  # rolling mean of last 3 = i-2

    def test_future_change_does_not_alter_past_features(self):
        y = np.arange(30, dtype=float); y2 = y.copy(); y2[-1] = 999
        d = pd.date_range("2021-01-01", periods=30, freq="MS")
        a, _ = build_supervised(y, d, [1, 2], [3]); b, _ = build_supervised(y2, d, [1, 2], [3])
        np.testing.assert_array_equal(a[:-1], b[:-1])

    def test_row_features_shape(self):
        self.assertEqual(len(row_features(np.arange(20.), "2024-05-01", [1, 2, 3], [3])), 3 + 2 + 5)


class TestModelsAndPipeline(unittest.TestCase):
    def test_baselines_exact(self):
        s = series_of(SC["seasonal"]); y = s.values
        r = run_model("Naive", s, 3, 12, "MS"); np.testing.assert_allclose(r["mean"], [y[-1]] * 3)
        r = run_model("Seasonal Naive", s, 14, 12, "MS"); np.testing.assert_allclose(r["mean"][:12], y[-12:]); self.assertEqual(r["mean"][12], y[-12])
        r = run_model("Simple Moving Average", s, 2, 12, "MS"); self.assertAlmostEqual(r["mean"][0], y[-3:].mean())

    def test_interval_ordering(self):
        s = series_of(SC["clean"])
        for m in ["Naive", "Seasonal Naive", "Simple Moving Average", "Random Forest", "Gradient Boosting"]:
            r = run_model(m, s, 6, 12, "MS"); self.assertTrue(np.all(r["lower"] <= r["mean"])); self.assertTrue(np.all(r["mean"] <= r["upper"]))

    def test_nan_training_rejected(self):
        with self.assertRaises(ValueError): run_model("Naive", series_of(SC["missing"]), 3, 12, "MS")

    def test_ml_follows_upward_trend(self):
        s = series_of(SC["upward_trend"]); r = run_model("Random Forest", s, 6, 12, "MS")
        self.assertGreater(r["mean"].mean(), s.iloc[-12:].mean())

    def test_seasonal_naive_beats_naive_on_seasonal(self):
        ev = evaluate_models(series_of(SC["seasonal"]), "MS", 12, PrepConfig(), ["Naive", "Seasonal Naive"], 6)
        h = ev["holdout"].set_index("Model"); self.assertLess(h.loc["Seasonal Naive", "RMSE"], h.loc["Naive", "RMSE"])
        self.assertEqual(ev["selected"], "Seasonal Naive")

    def test_evaluation_structure_and_chronology(self):
        ev = evaluate_models(series_of(SC["clean"]), "MS", 12, PrepConfig(), ["Naive", "Seasonal Naive", "Random Forest"], 6)
        self.assertLess(ev["train_period"][1], ev["test_period"][0])
        self.assertEqual(len(ev["predictions"]), ev["test_h"])
        self.assertEqual(ev["test_period"][1], series_of(SC["clean"]).index[-1])
        for f in ev["cv_folds"]: self.assertLess(f[1], f[2])
        self.assertEqual(ev["selected"], ev["cv"].iloc[0]["Model"])

    def test_failed_model_reported_not_hidden(self):
        ev = evaluate_models(series_of(SC["clean"]), "MS", 12, PrepConfig(), ["Naive", "Holt-Winters" if not HAS_SM else "Naive"], 6)
        self.assertIn("Naive", ev["cv"]["Model"].values)

    def test_missing_dataset_with_treatment(self):
        ev = evaluate_models(series_of(SC["missing"]), "MS", 12, PrepConfig(MISSING_METHODS[2]), ["Naive", "Seasonal Naive"], 6)
        self.assertFalse(ev["cv"].empty)

    def test_missing_without_treatment_fails_clearly(self):
        with self.assertRaises(ValueError): evaluate_models(series_of(SC["missing"]), "MS", 12, PrepConfig(MISSING_METHODS[0]), ["Naive"], 6)

    def test_future_forecast_dates(self):
        s = series_of(SC["clean"]); fc, meta, _ = future_forecast(s, "MS", 12, PrepConfig(), "Seasonal Naive", 6)
        self.assertEqual(fc["date"].iloc[0], s.index[-1] + pd.offsets.MonthBegin(1)); self.assertEqual(len(fc), 6)

    def test_sample_dataset(self):
        d = make_sample_sales(); self.assertGreaterEqual(d["Date"].nunique(), 24)
        self.assertEqual(validate_dataset(d, "Date", "Sales", "Category")["errors"], [])


@unittest.skipUnless(HAS_SM, "statsmodels not installed")
class TestStatsmodels(unittest.TestCase):
    def test_all_statsmodels(self):
        s = series_of(SC["seasonal"])
        for m in ["Simple Exp. Smoothing", "Holt Linear Trend", "Holt-Winters", "ARIMA", "SARIMA"]:
            r = run_model(m, s, 6, 12, "MS"); self.assertEqual(len(r["mean"]), 6); self.assertTrue(np.all(np.isfinite(r["mean"])))
            self.assertTrue(np.all(r["lower"] <= r["upper"]))

    def test_eda(self):
        from exploratory_analysis.eda import decompose, stationarity_tests
        s = series_of(SC["upward_trend"]); self.assertIn("Non", stationarity_tests(s)["verdict"] + "Non")
        self.assertEqual(len(decompose(s, 12)), 48)


if __name__ == "__main__":
    unittest.main()
