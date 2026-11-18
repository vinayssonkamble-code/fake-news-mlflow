
"""Metric, latency and bootstrap helpers using NumPy only."""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    """Confusion matrix: rows are actual, columns are predicted."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    matrix = np.zeros((2, 2), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        if actual in (0, 1) and predicted in (0, 1):
            matrix[actual, predicted] += 1

    return matrix


def compute_metrics(y_true, y_pred):
    """Binary metrics with FAKE (label 1) as the positive class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion(y_true, y_pred)

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    class_f1 = []

    for label in (0, 1):
        actual_positive = np.sum(y_true == label)
        predicted_positive = np.sum(y_pred == label)
        true_positive = np.sum(
            (y_true == label) & (y_pred == label)
        )

        p = true_positive / max(predicted_positive, 1)
        r = true_positive / max(actual_positive, 1)

        score = (
            2 * p * r / (p + r)
            if p + r > 0 else 0.0
        )

        class_f1.append(score)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(np.mean(class_f1)),
    }


def summarise_latency(seconds):
    s = np.asarray(seconds, dtype=float)
    ms = s * 1000.0

    return {
        "avg_latency_ms": float(ms.mean()),
        "p95_latency_ms": float(np.percentile(ms, 95)),
        "total_inference_s": float(s.sum()),
    }


def paired_bootstrap_delta(
    y_true, pred_a, pred_b, n_boot=2000, seed=42
):
    """Paired bootstrap of macro-F1 difference: B minus A."""
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)

    for k in range(n_boot):
        indices = rng.integers(0, n, n)

        score_a = compute_metrics(
            y[indices], a[indices]
        )["macro_f1"]

        score_b = compute_metrics(
            y[indices], b[indices]
        )["macro_f1"]

        deltas[k] = score_b - score_a

    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_boot": n_boot,
    }
