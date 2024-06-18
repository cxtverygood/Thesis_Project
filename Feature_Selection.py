import numpy as np


def feature_selection_mean(weights=None, sparsity=0.7):
    means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
    means_sorted = np.sort(means)
    threshold_idx = int(means.size * sparsity)

    n = len(means)
    if threshold_idx == n:
        return np.ones(n, dtype=bool)

    means_threshold = means_sorted[threshold_idx]

    feature_selection = means >= means_threshold

    return feature_selection
