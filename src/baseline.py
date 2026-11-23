"""Baselines avaliados nas MESMAS partições e métricas do modelo.

1) majoritario: sempre prevê a classe majoritária do treino.
2) hist_logreg: regressão logística sobre histograma de cor RGB (8 bins/canal). É um modelo simples
   que não usa redes pré-treinadas. Se ele ficar muito acima de 50%, desconfie de viés de captura
   (iluminação/fundo diferentes entre as classes), e não de "capacidade de ver dano".

Uso:
    python src/baseline.py --splits data/splits.csv --images data/raw --out outputs/baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import CLASS_TO_IDX, compute_metrics, jsonable


def hist_features(path: Path, bins: int = 8) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((224, 224))
    a = np.asarray(img).reshape(-1, 3)
    f = np.concatenate([np.histogram(a[:, c], bins=bins, range=(0, 256))[0] for c in range(3)]).astype(float)
    return f / f.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="outputs/baseline")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.splits)
    df["y"] = df["classe"].map(CLASS_TO_IDX)
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    y_train = parts["train"]["y"].values
    results = {"seed": a.seed, "n_train": len(parts["train"])}

    # 1) Classe majoritária (empate -> aprovada, a classe 0)
    maj = int(y_train.sum() > len(y_train) / 2)
    results["majoritario"] = {"classe_prevista": "recusada" if maj else "aprovada"}
    for s in ["val", "test"]:
        prob = np.full(len(parts[s]), float(maj))
        results["majoritario"][s] = compute_metrics(parts[s]["y"].values, prob)

    # 2) Histograma de cor + regressão logística (C fixo = 1.0, sem ajuste em teste)
    X = {s: np.stack([hist_features(Path(a.images) / n) for n in parts[s]["arquivo"]]) for s in parts}
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                             max_iter=2000, random_state=a.seed))
    clf.fit(X["train"], y_train)
    results["hist_logreg"] = {}
    for s in ["val", "test"]:
        prob = clf.predict_proba(X[s])[:, 1]
        results["hist_logreg"][s] = compute_metrics(parts[s]["y"].values, prob)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_results.json").write_text(json.dumps(jsonable(results), indent=2, ensure_ascii=False))
    for name in ["majoritario", "hist_logreg"]:
        for s in ["val", "test"]:
            m = results[name][s]
            print(f"{name:12s} {s:4s} n={m['n']:3d} acc={m['accuracy']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
