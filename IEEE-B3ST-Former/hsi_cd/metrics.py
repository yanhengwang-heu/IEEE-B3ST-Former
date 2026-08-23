from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    target = np.asarray(target, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if target.shape != pred.shape:
        raise ValueError(f"target/pred shape mismatch: {target.shape} vs {pred.shape}")

    tn = int(np.sum((target == 0) & (pred == 0)))
    fp = int(np.sum((target == 0) & (pred == 1)))
    fn = int(np.sum((target == 1) & (pred == 0)))
    tp = int(np.sum((target == 1) & (pred == 1)))
    total = max(tn + fp + fn + tp, 1)

    oa = (tp + tn) / total
    changed_acc = tp / max(tp + fn, 1)
    unchanged_acc = tn / max(tn + fp, 1)
    aa = 0.5 * (changed_acc + unchanged_acc)
    precision = tp / max(tp + fp, 1)
    recall = changed_acc
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    row0 = tn + fp
    row1 = fn + tp
    col0 = tn + fn
    col1 = fp + tp
    pe = (row0 * col0 + row1 * col1) / max(total * total, 1)
    kappa = (oa - pe) / max(1.0 - pe, 1e-12)

    return {
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "ChangedAcc": float(changed_acc),
        "UnchangedAcc": float(unchanged_acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Confusion": [[tn, fp], [fn, tp]],
    }

