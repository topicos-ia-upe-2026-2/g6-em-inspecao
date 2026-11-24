"""Valida data/metadata.csv + imagens e gera a divisão treino/validação/teste SEM vazamento.

A divisão é agrupada por `id_embalagem` (todas as fotos/ângulos/versões de uma mesma embalagem
física ficam no mesmo conjunto) e estratificada por classe.

Uso (a partir da raiz do repositório):
    python src/prepare_data.py --metadata data/metadata.csv --images data/raw --out data --seed 42

Saídas:
    data/splits.csv          arquivo, id_embalagem, tipo_embalagem, classe, split, largura, altura
    data/split_report.md     contagens por conjunto, classe e tipo de embalagem + checagem de vazamento
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED = ["arquivo", "id_embalagem", "tipo_embalagem", "classe",
            "tipo_dano", "origem_dano", "dispositivo", "responsavel"]
ALLOWED = {
    "classe": {"aprovada", "recusada"},
    "tipo_embalagem": {"plastico", "papel"},
    "origem_dano": {"controlado", "pre-existente", ""},  # vazio permitido para aprovadas
}


def load_and_validate(metadata_path: Path, images_dir: Path):
    # utf-8-sig: o metadata.csv do repositório começa com BOM
    df = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()

    errors, warnings = [], []
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        errors.append(f"Colunas ausentes em {metadata_path}: {missing_cols}")
        return df, errors, warnings
    if df.empty:
        errors.append(f"{metadata_path} não tem nenhuma linha de dados (ainda não houve coleta/registro).")
        return df, errors, warnings

    for col, allowed in ALLOWED.items():
        bad = df[~df[col].isin(allowed)]
        for _, r in bad.iterrows():
            errors.append(f"{r['arquivo']}: valor inválido em '{col}': {r[col]!r} (permitidos: {sorted(allowed)})")

    for col in ["arquivo", "id_embalagem"]:
        for _, r in df[df[col] == ""].iterrows():
            errors.append(f"linha com '{col}' vazio: {r.to_dict()}")

    dup = df[df.duplicated("arquivo", keep=False)]
    for name in dup["arquivo"].unique():
        errors.append(f"arquivo repetido no metadata: {name}")

    # Uma mesma embalagem física não pode ter tipo de material diferente
    for pid, g in df.groupby("id_embalagem"):
        if g["tipo_embalagem"].nunique() > 1:
            errors.append(f"id_embalagem {pid} aparece com mais de um tipo_embalagem")

    # Existência e integridade das imagens
    sizes = {}
    for name in df["arquivo"]:
        p = images_dir / name
        if not p.is_file():
            errors.append(f"imagem não encontrada: {p}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes[name] = im.size
        except (UnidentifiedImageError, OSError) as e:
            errors.append(f"imagem corrompida/ilegível: {p} ({e})")

    # Imagens na pasta sem linha no metadata
    if images_dir.is_dir():
        on_disk = {p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        orphans = sorted(on_disk - set(df["arquivo"]))
        if orphans:
            warnings.append(f"{len(orphans)} imagem(ns) na pasta sem linha no metadata (ignoradas): {orphans[:5]}...")

    df["largura"] = df["arquivo"].map(lambda n: sizes.get(n, (None, None))[0])
    df["altura"] = df["arquivo"].map(lambda n: sizes.get(n, (None, None))[1])

    # Equilíbrio (apenas aviso)
    share = (df["classe"] == "recusada").mean()
    if not 0.4 <= share <= 0.6:
        warnings.append(f"classes desbalanceadas: {share:.0%} recusada")
    ct = pd.crosstab(df["classe"], df["tipo_embalagem"])
    if "recusada" in ct.index:
        for tipo in ct.columns:
            tot = ct[tipo].sum()
            if tot and not 0.35 <= ct.loc["recusada", tipo] / tot <= 0.65:
                warnings.append(f"tipo '{tipo}' desbalanceado entre classes (risco de o modelo aprender o material)")
    return df, errors, warnings


def make_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """~70/15/15 agrupado por id_embalagem e estratificado por classe.

    1ª etapa: StratifiedGroupKFold(7) -> uma dobra (~1/7 ≈ 14%) vira TESTE.
    2ª etapa: nas restantes, StratifiedGroupKFold(6) -> uma dobra (~1/6 do restante ≈ 14%) vira VALIDAÇÃO.
    As proporções reais dependem do tamanho dos grupos e são reportadas em split_report.md.
    """
    y = (df["classe"] == "recusada").astype(int).values
    groups = df["id_embalagem"].values
    n_groups = df["id_embalagem"].nunique()
    if n_groups < 14:
        raise SystemExit(f"Poucas embalagens distintas ({n_groups}); são necessárias ao menos 14 para dividir em 3 conjuntos.")

    df = df.reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, y, groups))

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tv = df.iloc[trainval_idx]
    tr_rel, val_rel = next(inner.split(tv, y[trainval_idx], groups[trainval_idx]))

    df["split"] = ""
    df.loc[test_idx, "split"] = "test"
    df.loc[tv.index[val_rel], "split"] = "val"
    df.loc[tv.index[tr_rel], "split"] = "train"
    return df


def write_report(df: pd.DataFrame, warnings, path: Path, seed: int):
    lines = [f"# Relatório da divisão dos dados (seed={seed})", ""]
    n = len(df)
    lines += ["## Imagens por conjunto e classe", "",
              "| Conjunto | Imagens | % | aprovada | recusada | Embalagens (grupos) |", "|---|---|---|---|---|---|"]
    for s in ["train", "val", "test"]:
        d = df[df["split"] == s]
        lines.append(f"| {s} | {len(d)} | {len(d) / n:.1%} | {(d['classe'] == 'aprovada').sum()} | "
                     f"{(d['classe'] == 'recusada').sum()} | {d['id_embalagem'].nunique()} |")
    lines += ["", "## Tipo de embalagem por conjunto e classe", "",
              "| Conjunto | Classe | plastico | papel |", "|---|---|---|---|"]
    for s in ["train", "val", "test"]:
        for c in ["aprovada", "recusada"]:
            d = df[(df["split"] == s) & (df["classe"] == c)]
            lines.append(f"| {s} | {c} | {(d['tipo_embalagem'] == 'plastico').sum()} | {(d['tipo_embalagem'] == 'papel').sum()} |")

    leaks = df.groupby("id_embalagem")["split"].nunique()
    leaked = leaks[leaks > 1]
    lines += ["", "## Checagem de vazamento", "",
              f"- Embalagens em mais de um conjunto: **{len(leaked)}** (esperado: 0)"]
    if len(warnings):
        lines += ["", "## Avisos", ""] + [f"- {w}" for w in warnings]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(leaked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df, errors, warnings = load_and_validate(Path(a.metadata), Path(a.images))
    for w in warnings:
        print("AVISO:", w)
    if errors:
        print(f"\n{len(errors)} erro(s) de validação:")
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)

    df = make_splits(df, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["arquivo", "id_embalagem", "tipo_embalagem", "classe", "split", "largura", "altura"]
    df[cols].to_csv(out / "splits.csv", index=False)
    n_leaks = write_report(df, warnings, out / "split_report.md", a.seed)
    print((out / "split_report.md").read_text(encoding="utf-8"))
    if n_leaks:
        print("ERRO: vazamento detectado.")
        sys.exit(2)


if __name__ == "__main__":
    main()
