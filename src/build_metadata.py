"""Converte o metadata do conjunto 'bolachas_adriao_dataset_200' para o formato do projeto, rotulando POR FONTE.

Passo 1 (gera o modelo de rotulagem, classe em branco):
    python src/build_metadata.py init --orig data/raw/metadata.csv --fontes data/fontes.csv
Passo 2: preencher data/fontes.csv (classe, tipo_dano, origem_dano, tipo_embalagem, responsavel).
Passo 3 (gera data/metadata.csv completo):
    python src/build_metadata.py build --orig data/raw/metadata.csv --fontes data/fontes.csv --out data/metadata.csv
"""
import argparse
import sys
import pandas as pd  # type: ignore[reportMissingModuleSource]

FONTES_COLS = ["id_embalagem", "origem", "n_imagens", "tipo_embalagem", "classe", "tipo_dano", "origem_dano",
               "responsavel", "observacao"]


def add_source_ids(orig: pd.DataFrame) -> pd.DataFrame:
    """V1..Vn = cada vídeo; F01..Fn = cada foto original (ordenadas pelo nome de arquivo)."""
    orig = orig.copy()
    videos = sorted(orig.loc[orig["tipo"] == "frame_video", "origem"].unique())
    vmap = {o: f"V{i + 1}" for i, o in enumerate(videos)}
    fotos = orig[orig["tipo"] == "foto_original"].sort_values("arquivo")["arquivo"].tolist()
    fmap = {a: f"F{i + 1:02d}" for i, a in enumerate(fotos)}
    orig["id_embalagem"] = [vmap[o] if t == "frame_video" else fmap[a]
                            for a, o, t in zip(orig["arquivo"], orig["origem"], orig["tipo"])]
    return orig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["init", "build"])
    ap.add_argument("--orig", required=True)
    ap.add_argument("--fontes", default="data/fontes.csv")
    ap.add_argument("--out", default="data/metadata.csv")
    a = ap.parse_args()

    orig = pd.read_csv(a.orig, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    orig = add_source_ids(orig)

    if a.action == "init":
        g = orig.groupby("id_embalagem").agg(origem=("origem", "first"), n_imagens=("arquivo", "count")).reset_index()
        g["_k"] = g["id_embalagem"].str[0].map({"F": 0, "V": 1})
        g = g.sort_values(["_k", "id_embalagem"]).drop(columns="_k")
        g["tipo_embalagem"] = "plastico"   # confira: troque para 'papel' quando for o caso
        for c in ["classe", "tipo_dano", "origem_dano", "responsavel", "observacao"]:
            g[c] = ""
        g[FONTES_COLS].to_csv(a.fontes, index=False, encoding="utf-8-sig")
        print(f"{len(g)} fontes gravadas em {a.fontes}. Preencha ao menos 'classe' (aprovada|recusada).")
        return

    f = pd.read_csv(a.fontes, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    f["classe"] = f["classe"].str.strip().str.lower()
    pend = f[~f["classe"].isin(["aprovada", "recusada"])]
    if len(pend):
        print("Fontes sem classe válida (aprovada|recusada):", ", ".join(pend["id_embalagem"]))
        sys.exit(1)
    rec = f[f["classe"] == "recusada"]
    if rec.empty or (f["classe"] == "aprovada").sum() == 0:
        print("AVISO: só existe uma classe; não é possível treinar um classificador binário.")
    f["tipo_dano"] = f["tipo_dano"].where(f["classe"] == "recusada", "")
    sem_origem = rec[rec["origem_dano"].str.strip() == ""]
    if len(sem_origem):
        print("AVISO: recusadas sem origem_dano (controlado|pre-existente|sintetico):", ", ".join(sem_origem["id_embalagem"]))

    m = orig.merge(f[["id_embalagem", "tipo_embalagem", "classe", "tipo_dano", "origem_dano", "responsavel"]],
                   on="id_embalagem", how="left")
    m["dispositivo"] = "celular (WhatsApp)"
    cols = ["arquivo", "id_embalagem", "tipo_embalagem", "classe", "tipo_dano", "origem_dano", "dispositivo", "responsavel"]
    m[cols].to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(m)} linhas em {a.out}")
    print(m.groupby(["classe"]).agg(imagens=("arquivo", "count"), fontes=("id_embalagem", "nunique")))


if __name__ == "__main__":
    main()