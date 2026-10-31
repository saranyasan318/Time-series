FREQUENCIES = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS"}
SEASON_LENGTH = {"D": 7, "W": 52, "MS": 12, "QS": 4}
PERIOD_NAME = {"D": "day", "W": "week", "MS": "month", "QS": "quarter"}
MIN_OBSERVATIONS = {"D": 60, "W": 60, "MS": 24, "QS": 12}
MA_WINDOW = {"D": 7, "W": 4, "MS": 3, "QS": 2}
LAGS_BY_FREQ = {"D": [1, 2, 3, 7, 14], "W": [1, 2, 3, 4, 8], "MS": [1, 2, 3, 6, 12], "QS": [1, 2, 3, 4]}


HORIZONS = {
    "D": {"7 days": 7, "30 days": 30, "3 months (90 days)": 90, "6 months (180 days)": 180},
    "W": {"4 weeks": 4, "13 weeks (3 months)": 13, "26 weeks (6 months)": 26, "52 weeks (12 months)": 52},
    "MS": {"3 months": 3, "6 months": 6, "12 months": 12},
    "QS": {"2 quarters (6 months)": 2, "4 quarters (12 months)": 4},
}

MISSING_METHODS = [
    "Leave as missing (blocks modelling)",
    "Linear interpolation (gaps of 2 periods or fewer)",
    "Seasonal average (same position in cycle)",
    "Median fill",
    "Forward fill (1 period only)",
    "Linear interpolation (all gaps)",
]
DEFAULT_MISSING = "Linear interpolation (all gaps)"   # never leaves holes, so irregular/gappy uploads still model
OUTLIER_METHODS = ["Keep (flag only)", "Winsorise flagged points to IQR fences", "Replace flagged points with rolling median"]
DUPLICATE_AGG = ["sum", "mean", "last"]
INTERVAL_LEVELS = {"80%": 0.80, "90%": 0.90, "95%": 0.95}
CV_MODES = ["expanding", "rolling", "walk-forward"]
WALK_FORWARD_STEPS = 6          # one-step-ahead origins used by walk-forward validation
DEFAULT_PRIMARY_METRIC = "RMSE"  # ranking metric; any of MAE/RMSE/MAPE/sMAPE/WAPE/MASE works
RANK_METRICS = ["RMSE", "MAE", "MAPE", "sMAPE", "WAPE", "MASE"]
OUTLIER_THRESHOLDS = {"Strict (z>3.0)": 3.0, "Standard (z>3.5)": 3.5, "Lenient (z>5.0)": 5.0}
DATE_FORMATS = {"Auto-detect": None, "YYYY-MM-DD": "%Y-%m-%d", "DD/MM/YYYY": "%d/%m/%Y", "MM/DD/YYYY": "%m/%d/%Y",
                "YYYY-MM": "%Y-%m", "DD-MM-YYYY": "%d-%m-%Y"}

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"

THEME = {"navy": "#0B1B3A", "baby_blue": "#A9D6F5", "lavender": "#C9B8F0", "grey": "#F3F4F8", "white": "#FFFFFF",
         "accent": "#5B6EE1", "muted": "#6B7694"}


MULTI_SEASONS = {"D": [7, 365.25], "W": [52.18], "MS": [12], "QS": [4]}
ML_TREND_DAMPING = 0.98   # damping of the extrapolated linear trend in ML models (avoids runaway trends)

YEAR_PERIODS = {"D": 364, "W": 52, "MS": 12, "QS": 4}
