import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.pages_q3 import (
    SALES, forecast_week_13, make_backtest, validate_weekly_dates, demand_alert
)


class TestQ3(unittest.TestCase):
    def test_week_13_forecasts(self):
        naive, avg = forecast_week_13(SALES)
        self.assertEqual(naive, 215.0)
        self.assertAlmostEqual(avg, 201.6666667, places=5)

    def test_expanding_window_backtest_mae(self):
        bt = make_backtest(SALES)
        self.assertEqual(bt["Week"].tolist(), list(range(5, 13)))
        self.assertEqual(bt["Training weeks"].tolist(), list(range(4, 12)))
        self.assertAlmostEqual(bt["Naive absolute error"].mean(), 11.375)
        self.assertAlmostEqual(bt["3-week average absolute error"].mean(), 18.3333333333)

    def test_backtest_rejects_too_few_observations(self):
        with self.assertRaises(ValueError):
            make_backtest([1, 2, 3, 4])

    def test_forecast_rejects_negative_or_nonfinite_values(self):
        with self.assertRaises(ValueError):
            forecast_week_13([1, 2, -3])
        with self.assertRaises(ValueError):
            forecast_week_13([1, np.nan, 3])

    def test_duplicate_and_missing_week_detection(self):
        dates = ["2024-01-07", "2024-01-14", "2024-01-14", "2024-01-28"]
        duplicates, missing, invalid = validate_weekly_dates(dates, "W-SUN")
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(missing), 1)
        self.assertEqual(invalid, 0)

    def test_same_week_different_days_is_duplicate_week(self):
        dates = ["2024-01-02", "2024-01-06", "2024-01-14"]
        duplicates, missing, invalid = validate_weekly_dates(dates, "W-SUN")
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0], pd.Timestamp("2024-01-07"))
        self.assertEqual(invalid, 0)

    def test_invalid_and_blank_date_count(self):
        _, _, invalid = validate_weekly_dates(["2024-01-07", "not-a-date", None])
        self.assertEqual(invalid, 2)

    def test_empty_dates(self):
        duplicates, missing, invalid = validate_weekly_dates([])
        self.assertEqual(len(duplicates), 0)
        self.assertEqual(len(missing), 0)
        self.assertEqual(invalid, 0)

    def test_demand_alert(self):
        result = demand_alert(100, 120, [1, 2, 3, 4, 5, 6, 7, 8], 0.8)
        self.assertTrue(result["alert"])
        self.assertEqual(result["n_errors"], 8)

    def test_demand_alert_rejects_invalid_coverage(self):
        with self.assertRaises(ValueError):
            demand_alert(100, 110, [1, 2, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
