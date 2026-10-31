"""Choose candidate models from the dataset's length, frequency and seasonality.

The recommender decides which models are *worth fitting*; the final winner is still chosen by chronological
cross-validation, never by this list. Each decision carries a human-readable reason.
"""
from forecasting.models import MODEL_FAMILY, model_availability

SHORT, MEDIUM = "short", "medium"


def length_class(n, season):
    if n < max(2 * season + 2, 30):
        return "short"
    if n < max(4 * season, 60):
        return "medium"
    return "long"


def recommend_models(n, season, freq, strongly_seasonal=None, has_trend=None, all_positive=True):
    """Return (recommended: list[str], excluded: dict[name -> reason], length_class)."""
    avail = model_availability()
    cls = length_class(n, season)
    excluded = {}
    for name, (ok, why) in avail.items():
        if not ok:
            excluded[name] = why

    def drop(name, why):
        if name in avail and avail[name][0] and name not in excluded:
            excluded[name] = why

    seasonal_models = ["Seasonal Naive", "Holt-Winters", "Holt-Winters (damped)", "Holt-Winters (multiplicative)", "SARIMA"]
    if season < 2 or n < 2 * season + 2:
        for m in seasonal_models:
            drop(m, f"needs >= {2 * season + 2} observations for a seasonal period of {season} (have {n})")
    if not all_positive:
        drop("Holt-Winters (multiplicative)", "multiplicative seasonality needs strictly positive data")
    if season > 24:
        drop("SARIMA", f"seasonal period {season} is too long for SARIMA (max 24); use Harmonic Regression / Holt-Winters instead")
    if cls == "short":
        for m in ("Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost", "Prophet", "Auto ARIMA", "Harmonic Regression"):
            drop(m, f"history too short for a data-hungry model ({n} observations)")
    elif cls == "medium":
        for m in ("Prophet",):
            drop(m, "needs a longer history to be stable")
    if strongly_seasonal is False:
        drop("Holt-Winters (damped)", "no meaningful seasonality detected")
    rec = [m for m in avail if avail[m][0] and m not in excluded]
    return rec, excluded, cls


def group_by_family(names):
    out = {}
    for n in names:
        out.setdefault(MODEL_FAMILY.get(n, "Other"), []).append(n)
    return out
