"""Dataset, transformações e construção da MobileNetV2 (Transfer Learning / Fine-Tuning parcial)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from metrics import CLASS_TO_IDX

IMG_SIZE = 224
FINETUNE_FROM = 14  # blocos features[14:] são descongelados no Fine-Tuning parcial (config. inicial da E1)
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train: bool):
    # Redimensionamento direto (sem corte central) para não descartar danos nas bordas.
    base = [T.Resize((IMG_SIZE, IMG_SIZE))]
    aug = [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(brightness=0.2, contrast=0.2)] if train else []
    return T.Compose(base + aug + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PackagingDataset(Dataset):
    def __init__(self, splits_csv, images_dir, split, train_aug=False):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tf = get_transforms(train_aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = ImageOps.exif_transpose(Image.open(self.images_dir / r["arquivo"])).convert("RGB")
        return self.tf(img), CLASS_TO_IDX[r["classe"]], r["arquivo"]


def build_model(mode: str, pretrained: bool = True) -> nn.Module:
    """mode: 'transfer' (só o classificador treina) ou 'finetune' (features[14:] + classificador)."""
    assert mode in {"transfer", "finetune"}
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None  # explícito, nunca DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(1280, 2)  # mantém o Dropout existente
    for p in model.features.parameters():
        p.requires_grad = False
    if mode == "finetune":
        for p in model.features[FINETUNE_FROM:].parameters():
            p.requires_grad = True
    return model


def set_train_mode(model: nn.Module, mode: str):
    """model.train() + mantém em modo de avaliação (BatchNorm congelado) as camadas congeladas."""
    model.train()
    if mode == "transfer":
        model.features.eval()
    else:
        for i, block in enumerate(model.features):
            if i < FINETUNE_FROM:
                block.eval()


@torch.no_grad()
def run_eval(model, loader, device, criterion=None):
    """Retorna (perda média ou None, y_true, prob_recusada, arquivos)."""
    model.eval()
    ys, ps, names, tot, n = [], [], [], 0.0, 0
    for x, y, f in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            tot += criterion(logits, y).item() * len(y)
        n += len(y)
        ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
        names += list(f)
    return (tot / n if criterion is not None else None), np.concatenate(ys), np.concatenate(ps), names
