"""Avalia um checkpoint em val ou test e grava métricas (com IC95%), matriz de confusão e erros.

Uso:
    python src/evaluate.py --mode finetune --checkpoint outputs/finetune/best.pt --split val
    python src/evaluate.py --mode finetune --checkpoint outputs/finetune/best.pt --split test --threshold 0.5

--threshold val: escolhe o limiar que maximiza F1 NA VALIDAÇÃO e o aplica ao conjunto pedido.
Rode --split test uma única vez, depois de fixar a configuração (evita ajustar usando o teste).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import CLASSES, compute_metrics, jsonable
from model_utils import PackagingDataset, build_model, run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["transfer", "finetune"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--threshold", default="0.5", help="número em (0,1) ou 'val'")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.mode, pretrained=False).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))

    def probs(split):
        ds = PackagingDataset(a.splits, a.images, split)
        t = time.time()
        _, y, p, names = run_eval(model, DataLoader(ds, batch_size=16), device)
        return y, p, names, (time.time() - t) / max(len(ds), 1)

    if a.threshold == "val":
        yv, pv, _, _ = probs("val")
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        thr = float(max(grid, key=lambda t: (compute_metrics(yv, pv, t)["f1"], -abs(t - 0.5))))
    else:
        thr = float(a.threshold)

    y, p, names, sec_per_img = probs(a.split)
    m = compute_metrics(y, p, thr)
    m["segundos_por_imagem_(inclui_leitura_e_preprocessamento)"] = sec_per_img

    out = Path(a.out or Path(a.checkpoint).parent / f"avaliacao_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(jsonable(m), indent=2, ensure_ascii=False))
    pred = pd.DataFrame({"arquivo": names, "y_true": y, "prob_recusada": p, "y_pred": (p >= thr).astype(int)})
    pred["erro"] = pred["y_true"] != pred["y_pred"]
    pred.to_csv(out / "predictions.csv", index=False)

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(out / "matriz_confusao.png", dpi=120); plt.close(fig)

    ci = lambda k: f"[{m[k][0]:.2f}, {m[k][1]:.2f}]"
    print(f"{a.split} n={m['n']} limiar={thr:.2f} | acc={m['accuracy']:.3f} {ci('ci95_accuracy')} "
          f"recall={m['recall']:.3f} {ci('ci95_recall')} espec={m['specificity']:.3f} "
          f"prec={m['precision']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
