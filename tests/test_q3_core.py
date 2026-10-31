"""Tests for q3_core: hand-computed values, leakage guarantees, date edge cases, alert edge cases."""
import os, sys, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from q3_core import (SALES, METHODS, validate_values, forecast_next, expanding_backtest, backtest_summary,
                     select_baseline, leaky_neighbour_mae, audit_weekly_dates, validate_weekly_dates,
                     alert_tolerance, demand_alert, decision_memo)


class TestForecasts(unittest.TestCase):
    def test_week13(self):
        f = forecast_next(SALES)
        self.assertEqual(f["target_week"], 13)
        self.assertEqual(f[METHODS[0]], 215.0)
        self.assertAlmostEqual(f[METHODS[1]], (185 + 205 + 215) / 3)

    def test_minimum_length(self):
        self.assertEqual(forecast_next([1, 2, 3])[METHODS[1]], 2.0)
        with self.assertRaises(ValueError):
            forecast_next([1, 2])

    def test_bad_values(self):
        for bad in ([1, 2, -3], [1, np.nan, 3], [1, np.inf, 3], [1, "x", 3], [None, 1, 2]):
            with self.assertRaises(ValueError):
                validate_values(bad)


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.bt = expanding_backtest(SALES)

    def test_weeks_and_window_growth(self):
        self.assertEqual(self.bt["Week"].tolist(), list(range(5, 13)))
        self.assertEqual(self.bt["Training weeks"].tolist(), list(range(4, 12)))

    def test_hand_computed_first_row(self):
        r = self.bt.iloc[0]          # week 5 = 152, history 120,130,125,140
        self.assertEqual(r["Naive forecast"], 140)
        self.assertAlmostEqual(r["3-week average forecast"], (130 + 125 + 140) / 3)

    def test_mae(self):
        s = backtest_summary(self.bt)
        self.assertAlmostEqual(s["naive_mae"], 91 / 8)
        self.assertAlmostEqual(s["avg_mae"], 18.3333333333, places=6)
        self.assertEqual(select_baseline(s["naive_mae"], s["avg_mae"]), METHODS[0])

    def test_tie_goes_to_naive(self):
        self.assertEqual(select_baseline(5.0, 5.0), METHODS[0])

    def test_no_leakage_future_values_do_not_change_past_forecasts(self):
        """Changing week 12 must not change any forecast for weeks 5-11."""
        y2 = list(SALES); y2[-1] = 9999
        bt2 = expanding_backtest(y2)
        cols = ["Naive forecast", "3-week average forecast"]
        pd.testing.assert_frame_equal(self.bt.iloc[:-1][cols], bt2.iloc[:-1][cols])

    def test_shuffling_changes_result(self):
        """Order matters: a shuffled series gives different (and here worse) naive MAE."""
        rng = np.random.default_rng(0)
        y = np.array(SALES, float); rng.shuffle(y)
        shuffled = backtest_summary(expanding_backtest(y))["naive_mae"]
        self.assertNotAlmostEqual(shuffled, 11.375)

    def test_leaky_neighbour_demo_is_unrealistically_good(self):
        self.assertLess(leaky_neighbour_mae(SALES), backtest_summary(self.bt)["naive_mae"])

    def test_too_short_or_bad_start(self):
        with self.assertRaises(ValueError):
            expanding_backtest([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            expanding_backtest(SALES, first_test_week=3)

    def test_minimum_series_gives_one_row(self):
        self.assertEqual(len(expanding_backtest([10, 11, 12, 13, 14])), 1)

    def test_constant_series_zero_error(self):
        s = backtest_summary(expanding_backtest([7] * 9))
        self.assertEqual(s["naive_mae"], 0.0); self.assertEqual(s["avg_mae"], 0.0)


class TestDates(unittest.TestCase):
    def test_missing_and_duplicate(self):
        r = audit_weekly_dates(["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-28"])
        self.assertEqual([str(d.date()) for d in r["exact_duplicates"]], ["2024-01-14"])
        self.assertEqual([str(d.date()) for d in r["missing_weeks"]], ["2024-01-21"])
        self.assertFalse(r["ok"])

    def test_clean_series(self):
        r = audit_weekly_dates(pd.date_range("2024-01-07", periods=12, freq="W-SUN"))
        self.assertTrue(r["ok"]); self.assertEqual(r["missing_weeks"], [])

    def test_same_week_different_days(self):
        r = audit_weekly_dates(["2024-01-02", "2024-01-06", "2024-01-14"])
        self.assertEqual(r["exact_duplicates"], [])
        self.assertEqual(len(r["same_week_duplicates"]), 1)

    def test_invalid_blank_and_unsorted(self):
        r = audit_weekly_dates(["2024-01-14", "nope", None, "2024-01-07"])
        self.assertEqual(r["invalid"], 2); self.assertTrue(r["out_of_order"])

    def test_empty_and_all_invalid(self):
        self.assertFalse(audit_weekly_dates([])["ok"])
        self.assertFalse(audit_weekly_dates(["x", "y"])["ok"])

    def test_monday_convention(self):
        r = audit_weekly_dates(pd.date_range("2024-01-01", periods=5, freq="W-MON"), freq="W-MON")
        self.assertTrue(r["ok"])

    def test_no_timestamps_is_labelled_assumption(self):
        r = audit_weekly_dates(None, n_obs=12, assumed_start="2024-01-07")
        self.assertIsNone(r["ok"]); self.assertIn("ASSUMPTION", r["note"])
        self.assertEqual(len(r["assumed_index"]), 12)
        with self.assertRaises(ValueError):
            audit_weekly_dates(None)

    def test_bad_freq(self):
        with self.assertRaises(ValueError):
            audit_weekly_dates(["2024-01-07"], freq="D")

    def test_legacy_wrapper(self):
        dup, miss, inv = validate_weekly_dates(["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-28"])
        self.assertEqual((len(dup), len(miss), inv), (1, 1, 0))
        d, m, i = validate_weekly_dates([])
        self.assertEqual((len(d), len(m), i), (0, 0, 0))


class TestAlert(unittest.TestCase):
    ERR = [12, 3, 16, 5, 20, 5, 20, 10]

    def test_band_80(self):
        t = alert_tolerance(self.ERR, 0.8)
        self.assertEqual(t["half_width"], 20.0); self.assertTrue(t["guarantee_available"])

    def test_90_has_no_guarantee_with_8_errors(self):
        t = alert_tolerance(self.ERR, 0.9)
        self.assertFalse(t["guarantee_available"]); self.assertEqual(t["min_errors_needed"], 10)

    def test_direction(self):
        self.assertEqual(demand_alert(215, 250, self.ERR)["direction"], "above")
        self.assertEqual(demand_alert(215, 150, self.ERR)["direction"], "below")
        a = demand_alert(215, 230, self.ERR)
        self.assertFalse(a["alert"]); self.assertEqual(a["direction"], "within")

    def test_boundary_not_alert(self):
        self.assertFalse(demand_alert(215, 235, self.ERR)["alert"])   # exactly on the edge

    def test_lower_bound_clipped_at_zero(self):
        self.assertEqual(demand_alert(5, 0, self.ERR)["lower"], 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): alert_tolerance([], 0.8)
        with self.assertRaises(ValueError): alert_tolerance([np.nan], 0.8)
        with self.assertRaises(ValueError): alert_tolerance(self.ERR, 1.0)
        with self.assertRaises(ValueError): demand_alert(-1, 5, self.ERR)
        with self.assertRaises(ValueError): demand_alert(5, np.nan, self.ERR)


class TestMemo(unittest.TestCase):
    def test_memo_numbers_come_from_computation(self):
        m = decision_memo(SALES)
        for needle in ("11.375", "18.333", "215.0", "201.7", "Naive last value"):
            self.assertIn(needle, m)
        self.assertNotIn("outperform", m.replace("not evidence", ""))   # no accuracy promise

    def test_memo_follows_data_not_a_fixed_answer(self):
        flat_noisy = [100, 120, 90, 110, 95, 115, 100, 112, 98, 108, 102, 110]
        m = decision_memo(flat_noisy)
        s = backtest_summary(expanding_backtest(flat_noisy))
        self.assertIn(select_baseline(s["naive_mae"], s["avg_mae"]), m)


if __name__ == "__main__":
    unittest.main()
