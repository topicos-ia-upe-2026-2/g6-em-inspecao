"""Treino da MobileNetV2: Transfer Learning (--mode transfer) ou Fine-Tuning parcial (--mode finetune).

Uso (raiz do repositório):
    python src/train.py --mode transfer
    python src/train.py --mode finetune
Saídas em outputs/<mode>/: log_treino.csv, curvas.png, run_config.json, val_predictions.csv, best.pt (fora do Git).
O conjunto de TESTE não é usado aqui; use src/evaluate.py uma única vez após fixar a configuração.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import compute_metrics, jsonable
from model_utils import PackagingDataset, build_model, run_eval, set_seed, set_train_mode


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["transfer", "finetune"], required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default=None, help="padrão: outputs/<mode>")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="padrão: 1e-3 (transfer) | 1e-4 (finetune)")
    ap.add_argument("--patience", type=int, default=5, help="early stopping na perda de validação")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="SOMENTE para teste de fumaça do código")
    a = ap.parse_args()

    lr = a.lr if a.lr is not None else (1e-3 if a.mode == "transfer" else 1e-4)
    out = Path(a.out or f"outputs/{a.mode}")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = PackagingDataset(a.splits, a.images, "train", train_aug=True)
    va = PackagingDataset(a.splits, a.images, "val")
    g = torch.Generator().manual_seed(a.seed)
    tr_loader = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, generator=g)
    va_loader = DataLoader(va, batch_size=a.batch_size, num_workers=a.num_workers)

    model = build_model(a.mode, pretrained=not a.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    log, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        te = time.time()
        set_train_mode(model, a.mode)
        run_loss, n = 0.0, 0
        for x, y, _ in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        val_loss, yt, yp, _ = run_eval(model, va_loader, device, criterion)
        m = compute_metrics(yt, yp)
        row = {"epoca": epoch, "train_loss": run_loss / n, "val_loss": val_loss, "val_acc": m["accuracy"],
               "val_recall": m["recall"], "val_f1": m["f1"], "segundos": round(time.time() - te, 1)}
        log.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping na época {epoch} (melhor: {best_epoch})")
                break

    total_s = time.time() - t0
    pd.DataFrame(log).to_csv(out / "log_treino.csv", index=False)

    # Curvas
    d = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(d["epoca"], d["train_loss"], label="treino")
    ax[0].plot(d["epoca"], d["val_loss"], label="validação")
    ax[0].axvline(best_epoch, ls="--", c="gray")
    ax[0].set_xlabel("época"); ax[0].set_ylabel("perda"); ax[0].legend()
    ax[1].plot(d["epoca"], d["val_f1"], label="F1 (val)")
    ax[1].plot(d["epoca"], d["val_recall"], label="recall recusada (val)")
    ax[1].set_xlabel("época"); ax[1].legend()
    fig.suptitle(f"MobileNetV2 – {a.mode}")
    fig.tight_layout(); fig.savefig(out / "curvas.png", dpi=120); plt.close(fig)

    # Predições de validação do MELHOR checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    _, yt, yp, names = run_eval(model, va_loader, device)
    pd.DataFrame({"arquivo": names, "y_true": yt, "prob_recusada": yp}).to_csv(out / "val_predictions.csv", index=False)
    best_metrics = compute_metrics(yt, yp)

    cfg = {
        "mode": a.mode, "args": vars(a), "lr_efetivo": lr, "otimizador": "Adam", "perda": "CrossEntropyLoss",
        "pesos": None if a.no_pretrained else "MobileNet_V2_Weights.IMAGENET1K_V2",
        "parametros_treinaveis": n_train_params, "parametros_total": n_total,
        "n_treino": len(tr), "n_validacao": len(va),
        "melhor_epoca": best_epoch, "epocas_executadas": len(log), "melhor_val_loss": best_loss,
        "tempo_total_s": round(total_s, 1), "metricas_val_melhor_checkpoint": best_metrics,
        "ambiente": {"python": platform.python_version(), "torch": torch.__version__,
                     "plataforma": platform.platform(), "dispositivo": str(device),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "git_commit": git_commit(),
    }
    (out / "run_config.json").write_text(json.dumps(jsonable(cfg), indent=2, ensure_ascii=False))
    print(f"\nMelhor época: {best_epoch} | val F1={best_metrics['f1']:.3f} recall={best_metrics['recall']:.3f} | {total_s:.0f}s")


if __name__ == "__main__":
    main()
