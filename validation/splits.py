
def chronological_split(s, test_size):
    if not 0 < test_size < len(s):
        raise ValueError("test_size must be between 1 and len(series)-1")
    return s.iloc[:-test_size], s.iloc[-test_size:]


def time_series_folds(n, test_size, n_splits, min_train, mode="expanding", window=None):

    if mode == "walk-forward":
        test_size = 1
    folds = []
    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        if test_start < min_train:
            continue
        start = 0 if mode in ("expanding", "walk-forward") else max(0, test_start - (window or test_start))
        folds.append((start, test_start, test_end))
    return folds


def assert_no_leakage(train, test):
    if train.index.max() >= test.index.min():
        raise AssertionError("Leakage: training data overlaps or follows the test period")
    return True
