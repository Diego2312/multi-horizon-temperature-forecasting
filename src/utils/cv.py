import numpy as np

def rolling_origin_splits(n, n_splits=5, test_size=60):
    for i in range(n_splits):
        train_end = n - (n_splits - i) * test_size
        if train_end <= 0:
            continue
        train_idx = np.arange(train_end)
        test_idx = np.arange(train_end, min(n, train_end + test_size))
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx
