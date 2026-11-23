"""Métricas de classificação binária. Classe positiva = RECUSADA (1)."""
import math

import numpy as np
from sklearn.metrics import roc_auc_score

CLASSES = ["aprovada", "recusada"]  # índice 0 = aprovada, 1 = recusada (positiva)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Intervalo de confiança de Wilson (95%) para uma proporção k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ratio(num, den):
    return num / den if den else float("nan")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_prob = probabilidade da classe RECUSADA. Retorna contagens, métricas e IC95% (Wilson)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _ratio(tp + tn, n)

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "n": n, "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "roc_auc": auc,
        "ci95_accuracy": wilson_interval(tp + tn, n),
        "ci95_recall": wilson_interval(tp, tp + fn),
        "ci95_specificity": wilson_interval(tn, tn + fp),
        "ci95_precision": wilson_interval(tp, tp + fp),
    }


def jsonable(obj):
    """Converte NaN em None e tuplas em listas para salvar JSON válido."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
