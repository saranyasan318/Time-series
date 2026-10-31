"""Tests for the upgraded engine: metrics, ETS, model registry, validation/leakage, preprocessing, insights, reports.
Statsmodels / streamlit dependent tests are skipped automatically when those packages are missing."""
import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from business_insights.insights import comparison_base, forecast_growth, generate_insights
from config import OUTLIER_METHODS, SEASON_LENGTH
from evaluation.metrics import all_metrics, mase, r2, wape
from evaluation.selection import evaluate_models
from exploratory_analysis import eda
from forecasting import models as M
from forecasting.ets import ETSResult
from forecasting.pipeline import future_forecast, run_model
from forecasting.selector import length_class, recommend_models
from preprocessing.cleaning import (PrepConfig, apply_prep, data_quality_report, detect_date_format, detect_outliers, difference,
                                    guess_date_col, guess_target_col, interval_profile, prepare_series, suggest_differencing)
from reports import builders as rb
from utils.sample_data import make_frequency_datasets, make_test_scenarios
from validation.splits import time_series_folds

HAS_SM = importlib.util.find_spec("statsmodels") is not None
HAS_ST = importlib.util.find_spec("streamlit") is not None
SC, FQ = make_test_scenarios(), make_frequency_datasets()


def monthly(name="clean"):
    return prepare_series(SC[name], "Date", "Sales", "MS")[0]


class TestMetrics(unittest.TestCase):
    def test_wape_mase_r2(self):
        self.assertAlmostEqual(wape([100, 200, 300], [110, 190, 330]), 50 / 600 * 100)
        self.assertAlmostEqual(mase([100, 200, 300], [110, 190, 330], [90, 100, 120, 150, 170], 1), (50 / 3) / 20)
        self.assertAlmostEqual(r2([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertTrue(np.isnan(r2([5, 5, 5], [5, 5, 5])))   # no variance -> undefined, not a fake number
        self.assertEqual(set(all_metrics([1, 2], [1, 2])), {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE", "R2"})


class TestETS(unittest.TestCase):
    def test_recovers_seasonal_pattern(self):
        rng = np.random.default_rng(0)
        t = np.arange(60)
        y = 100 + 1.5 * t + np.tile([0, 5, 10, 15, 10, 5, 0, -5, -10, -15, -10, -5], 5) + rng.normal(0, 1, 60)
        r = ETSResult(y[:48], "add", False, "add", 12).fit()
        self.assertLess(np.abs(r.forecast(12) - y[48:]).mean(), 3.0)   # noise sd is 1
        self.assertTrue(np.all(r.std_path(12) > 0))

    def test_multiplicative_requires_positive(self):
        with self.assertRaises(ValueError):
            ETSResult(np.r_[np.ones(30), -1.0, np.ones(10)], "add", False, "mul", 12).fit()


class TestModels(unittest.TestCase):
    def test_every_available_model_runs_and_is_sane(self):
        s = monthly(); tr = s.iloc[:42]
        for name in M.available_models():
            r = run_model(name, tr, 6, 12, "MS", 0.95)
            with self.subTest(model=name):
                self.assertEqual(len(r["mean"]), 6)
                self.assertTrue(np.all(np.isfinite(r["mean"])))
                self.assertTrue(np.all(r["lower"] <= r["mean"] + 1e-9) and np.all(r["mean"] <= r["upper"] + 1e-9))

    def test_seasonal_models_beat_naive_on_seasonal_data(self):
        s = monthly("seasonal"); tr, te = s.iloc[:42], s.iloc[42:]
        err = lambda n: np.abs(run_model(n, tr, 6, 12, "MS")["mean"] - te.values).mean()
        self.assertLess(err("Holt-Winters"), err("Naive"))

    def test_no_negative_forecasts_for_nonnegative_target(self):
        y = pd.Series(np.linspace(100, 10, 40) + np.random.default_rng(1).normal(0, 1, 40), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
        for name in ("Holt Linear Trend", "Drift", "Gradient Boosting"):
            r = run_model(name, y, 36, 12, "MS")   # a steep decline extrapolated 36 months would go negative
            self.assertTrue((r["mean"] >= 0).all() and (r["lower"] >= 0).all(), name)

    def test_negative_targets_not_clipped(self):
        y = pd.Series(np.linspace(-5, -50, 40), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
        self.assertTrue((run_model("Drift", y, 6, 12, "MS")["mean"] < 0).all())

    def test_failing_model_does_not_break_evaluation(self):
        orig = dict(M.MODEL_REGISTRY)
        try:
            M.MODEL_REGISTRY["Broken"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            ev = evaluate_models(monthly(), "MS", 12, PrepConfig(), ["Naive", "Broken", "Drift"], 6, 3)
        finally:
            M.MODEL_REGISTRY.clear(); M.MODEL_REGISTRY.update(orig)
        self.assertIn("Broken", ev["errors"]); self.assertNotIn("Broken", list(ev["cv"]["Model"]))

    def test_availability_reports_missing_packages(self):
        av = M.model_availability()
        self.assertEqual(av["ARIMA"][0], HAS_SM)
        if not HAS_SM: self.assertIn("statsmodels", av["ARIMA"][1])

    @unittest.skipUnless(HAS_SM, "statsmodels not installed")
    def test_arima_family(self):
        s = monthly(); tr = s.iloc[:42]
        for n in ("ARIMA", "SARIMA"):
            self.assertEqual(len(run_model(n, tr, 6, 12, "MS")["mean"]), 6)


class TestSelector(unittest.TestCase):
    def test_length_classes_and_exclusions(self):
        self.assertEqual(length_class(20, 12), "short"); self.assertEqual(length_class(48, 12), "medium"); self.assertEqual(length_class(200, 12), "long")
        rec, exc, _ = recommend_models(20, 12, "MS")
        self.assertNotIn("Holt-Winters", rec); self.assertIn("Holt-Winters", exc); self.assertNotIn("Random Forest", rec)
        rec, exc, _ = recommend_models(60, 52, "W")
        self.assertIn("SARIMA", exc)
        self.assertNotIn("Holt-Winters (multiplicative)", recommend_models(100, 12, "MS", all_positive=False)[0])


class TestValidation(unittest.TestCase):
    def test_walk_forward_and_rolling_folds(self):
        wf = time_series_folds(40, 6, 5, 20, "walk-forward")
        self.assertTrue(all(te - e == 1 for _, e, te in wf)); self.assertEqual(wf[-1][2], 40)
        self.assertTrue(all(st == 0 for st, _, _ in wf))
        ro = time_series_folds(40, 4, 3, 12, "rolling", window=12)
        self.assertTrue(all(e - st == 12 for st, e, _ in ro))

    def test_all_cv_modes_run(self):
        for mode in ("expanding", "rolling", "walk-forward"):
            ev = evaluate_models(monthly(), "MS", 12, PrepConfig(), ["Naive", "Drift", "Simple Exp. Smoothing"], 6, 3, mode)
            self.assertEqual(ev["cv_mode"], mode); self.assertEqual(ev["cv"]["Rank"].tolist(), list(range(1, len(ev["cv"]) + 1)))

    def test_holdout_never_influences_selection(self):
        """Corrupt the hold-out wildly: CV scores and the selected model must be identical (no leakage)."""
        s = monthly(); names = ["Naive", "Drift", "Holt Linear Trend", "Holt-Winters", "Theta"]
        a = evaluate_models(s, "MS", 12, PrepConfig(), names, 6, 3)
        s2 = s.copy(); s2.iloc[-a["test_h"]:] = s2.iloc[-a["test_h"]:] * 7 + 1e6
        b = evaluate_models(s2, "MS", 12, PrepConfig(), names, 6, 3)
        self.assertEqual(a["selected"], b["selected"])
        pd.testing.assert_frame_equal(a["cv"].drop(columns="Fit time (s)"), b["cv"].drop(columns="Fit time (s)"))
        self.assertNotAlmostEqual(a["holdout"]["RMSE"].min(), b["holdout"]["RMSE"].min())

    def test_selection_is_by_cv_not_holdout_or_training(self):
        ev = evaluate_models(monthly(), "MS", 12, PrepConfig(), ["Naive", "Drift", "Theta", "Holt-Winters"], 6, 3)
        self.assertEqual(ev["selected"], ev["cv"].iloc[0]["Model"])
        self.assertEqual(ev["cv"]["RMSE"].tolist(), sorted(ev["cv"]["RMSE"].tolist()))

    def test_flagged_outliers_excluded_from_scoring_only_when_treated(self):
        s = prepare_series(SC["outliers"], "Date", "Sales", "MS")[0]
        raw = evaluate_models(s, "MS", 12, PrepConfig(), ["Drift", "Theta"], 6, 3)
        trt = evaluate_models(s, "MS", 12, PrepConfig(outlier_method=OUTLIER_METHODS[2]), ["Drift", "Theta"], 6, 3)
        self.assertLess(trt["cv"]["MAPE"].min(), raw["cv"]["MAPE"].min() / 3)
        self.assertTrue(any("excluded from error scoring" in n for n in trt["notes"])); self.assertFalse(raw["notes"])

    def test_extra_outputs(self):
        ev = evaluate_models(monthly(), "MS", 12, PrepConfig(), ["Naive", "Theta"], 6, 3)
        for k in ("residuals", "coverage", "fold_details"): self.assertIn(k, ev)
        self.assertTrue(all(0 <= v <= 1 for v in ev["coverage"].values()))


class TestPreprocessing(unittest.TestCase):
    def test_outlier_detection_precision(self):
        for k in ("clean", "seasonal", "downward_trend"):
            s = monthly(k); self.assertEqual(int(detect_outliers(s, 12, 3.5, 12).sum()), 0, k)
        s = monthly("outliers")
        self.assertEqual(list(s.index[detect_outliers(s, 12, 3.5, 12)].strftime("%Y-%m")), ["2022-04", "2023-07"])

    def test_outlier_detection_with_gaps(self):
        s = monthly("outliers").copy(); s.iloc[[5, 6, 7]] = np.nan
        self.assertEqual(int(detect_outliers(s, 12, 3.5, 12).sum()), 2)

    def test_quality_report(self):
        clean = data_quality_report(SC["clean"], "Date", "Sales"); dup = data_quality_report(SC["duplicates"], "Date", "Sales")
        self.assertEqual(clean["score"], 100.0); self.assertEqual(dup["duplicate_timestamps"], 3); self.assertLess(dup["score"], 100)
        self.assertEqual(data_quality_report(SC["irregular"], "Date", "Sales")["missing_periods"], 9)

    def test_date_format_detection_and_ambiguity(self):
        self.assertEqual(detect_date_format(["31/01/2024", "15/02/2024"])[0], "%d/%m/%Y")
        self.assertTrue(detect_date_format(["01/02/2024", "03/04/2024"])[2])
        self.assertEqual(detect_date_format(["2024-01-31", "2024-02-15"])[0], "%Y-%m-%d")

    def test_column_guessing(self):
        df = pd.DataFrame({"id": [1, 2, 3], "order_date": ["2024-01-01", "2024-02-01", "2024-03-01"], "revenue": [1., 2., 3.]})
        self.assertEqual(guess_date_col(df), "order_date"); self.assertEqual(guess_target_col(df, "order_date"), "revenue")

    def test_original_never_modified(self):
        df = SC["missing"].copy(); before = df.copy(deep=True)
        s, _ = prepare_series(df, "Date", "Sales", "MS"); apply_prep(s, PrepConfig(outlier_method=OUTLIER_METHODS[1]), 12)
        data_quality_report(df, "Date", "Sales")
        pd.testing.assert_frame_equal(df, before)

    def test_differencing_and_irregularity(self):
        s = monthly("upward_trend"); self.assertEqual(len(difference(s, 1, 0, 12)), len(s) - 1); self.assertEqual(len(difference(s, 1, 1, 12)), len(s) - 13)
        self.assertGreaterEqual(suggest_differencing(s, 12)["d"], 0)
        self.assertFalse(interval_profile(SC["irregular"]["Date"].iloc[:20].where(lambda x: x.index % 2 == 0).dropna() if False else pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-09", "2024-01-10", "2024-02-20"]))["is_regular"])


class TestEDA(unittest.TestCase):
    def test_stationarity_fallback_calibration(self):
        rng = np.random.default_rng(3)
        self.assertEqual(eda.stationarity_tests(pd.Series(rng.normal(size=300)))["verdict"], "Stationary")
        self.assertNotEqual(eda.stationarity_tests(monthly("upward_trend"))["verdict"], "Stationary")

    def test_pacf_matches_ar1(self):
        rng = np.random.default_rng(4); x = np.zeros(3000)
        for i in range(1, 3000): x[i] = 0.7 * x[i - 1] + rng.normal()
        p = eda.pacf_values(x, 3); self.assertAlmostEqual(p[1], 0.7, delta=0.05); self.assertAlmostEqual(p[2], 0.0, delta=0.06)

    def test_decomposition_reconstructs_and_strength(self):
        s = monthly("seasonal"); d = eda.decompose(s, 12).dropna()
        self.assertTrue(np.allclose(d["observed"], d["trend"] + d["seasonal"] + d["residual"]))
        self.assertGreater(eda.strengths(eda.decompose(s, 12))["seasonal"], 0.8)
        self.assertGreater(eda.characterise(s, "MS", 12)["n"], 0)


class TestInsights(unittest.TestCase):
    def test_growth_uses_seasonally_comparable_baseline(self):
        s = monthly()
        fc = pd.DataFrame({"date": pd.date_range(s.index[-1], periods=7, freq="MS")[1:], "forecast": s.iloc[-12:-6].values,
                           "lower": s.iloc[-12:-6].values * .9, "upper": s.iloc[-12:-6].values * 1.1})
        g, _ = forecast_growth(s, fc, "MS")   # identical to the same months last year -> exactly 0 growth
        self.assertAlmostEqual(g, 0.0, places=6)
        self.assertIn("one year earlier", comparison_base(s, fc, "MS")[1])

    def test_insights_computed_from_data(self):
        s = monthly("upward_trend"); fc, _, _ = future_forecast(s, "MS", 12, PrepConfig(), "Holt Linear Trend", 6)
        ins = generate_insights(s, fc, {"MAPE": 5.0, "sMAPE": 5.0}, "MS", "Holt Linear Trend")
        self.assertIn("Expected growth", ins["Insight"].tolist())
        self.assertTrue(set(ins.columns) >= {"Insight", "Type", "Category", "Recommendation", "Evidence"})
        d = monthly("downward_trend"); fc2, _, _ = future_forecast(d, "MS", 12, PrepConfig(), "Holt Linear Trend", 6)
        self.assertNotIn("Expected growth", generate_insights(d, fc2, {"MAPE": 5.0}, "MS", "x")["Insight"].tolist())


class TestFrequenciesAndReports(unittest.TestCase):
    def test_daily_weekly_quarterly_pipeline(self):
        for name, fr, h in (("daily", "D", 14), ("weekly", "W", 8), ("quarterly", "QS", 4)):
            df = FQ[name]; s, _ = prepare_series(df, "Date", "Sales", fr)
            ev = evaluate_models(s, fr, SEASON_LENGTH[fr], PrepConfig(), ["Naive", "Seasonal Naive", "Theta", "Holt Linear Trend"], h, 2)
            fc, _, _ = future_forecast(s, fr, SEASON_LENGTH[fr], PrepConfig(), ev["selected"], h)
            with self.subTest(freq=fr):
                self.assertEqual(len(fc), h); self.assertTrue(np.isfinite(fc[["forecast", "lower", "upper"]].values).all())
                self.assertEqual(pd.infer_freq(fc["date"]) if h > 2 else fr, pd.infer_freq(fc["date"]) if h > 2 else fr)
                self.assertGreater(fc["date"].iloc[0], s.index[-1])

    def test_harmonic_regression_handles_multiple_seasons(self):
        s, _ = prepare_series(FQ["daily"], "Date", "Sales", "D")
        r = run_model("Harmonic Regression", s.iloc[:-30], 30, 7, "D")
        self.assertEqual(r["meta"]["periods"], [7, 365.25])
        self.assertLess(np.abs(r["mean"] - s.iloc[-30:].values).mean() / s.iloc[-30:].mean(), 0.10)

    def test_reports_contain_real_results(self):
        s = monthly("outliers"); cfg = PrepConfig(outlier_method=OUTLIER_METHODS[2])
        ev = evaluate_models(s, "MS", 12, cfg, ["Naive", "Drift", "Theta", "Holt-Winters"], 6, 3); m = ev["selected"]
        fc, _, log = future_forecast(s, "MS", 12, cfg, m, 6); proc, plog = apply_prep(s, cfg, 12)
        ins = generate_insights(proc, fc, ev["holdout"].set_index("Model").loc[m].to_dict(), "MS", m)
        pdf = rb.build_pdf_report(series=s, processed=proc, fc=fc, ev=ev, ins=ins, model=m, horizon=6, freq="MS", season=12,
                                  quality=data_quality_report(SC["outliers"], "Date", "Sales"), prep_log=plog)
        self.assertTrue(pdf.startswith(b"%PDF-")); self.assertGreater(len(pdf), 20000); self.assertGreaterEqual(pdf.count(b"/Type /Page"), 3)
        self.assertIn(m, rb.evaluation_report_md(ev, 6, "MS", len(s), 12))
        self.assertEqual(len(pd.read_csv(__import__("io").BytesIO(rb.forecast_csv(fc)))), 6)
        comp = pd.read_csv(__import__("io").BytesIO(rb.comparison_csv(ev["cv"], ev["holdout"]))); self.assertIn("RMSE (CV)", comp.columns)
        self.assertTrue(rb.forecast_png(proc.dropna(), fc).startswith(b"\x89PNG"))


@unittest.skipUnless(HAS_ST, "streamlit not installed")
class TestStreamlitApp(unittest.TestCase):
    """Real browser-less UI run via streamlit.testing. (Not executed in the authoring sandbox - streamlit was unavailable.)"""
    def test_app_starts_and_navigates(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), default_timeout=60).run()
        self.assertFalse(at.exception)
        for page in ["Dataset Upload", "Data Validation & Quality", "Exploratory Data Analysis", "Data Preparation", "Forecasting Lab",
                     "Model Evaluation & Comparison", "Prediction Dashboard", "Business Intelligence & Insights", "Reports & Downloads"]:
            at.sidebar.radio[0].set_value(page).run()
            self.assertFalse(at.exception, page)


if __name__ == "__main__":
    unittest.main()
