# Experimentos

Preencher **somente com valores medidos**, copiados dos arquivos indicados. Não registrar estimativas.

## Execuções

| ID | Modo | Comando | Commit | Ambiente | Épocas (melhor/total) | Tempo | Evidências |
|---|---|---|---|---|---|---|---|
| E2-transfer | transfer | `python src/train.py --mode transfer` | _(run_config.json → git_commit)_ | _(run_config.json → ambiente)_ | _ | _ | `outputs/transfer/` |
| E2-finetune | finetune | `python src/train.py --mode finetune` | _ | _ | _ | _ | `outputs/finetune/` |

## Resultados na validação (E2)

Fonte: `outputs/<modo>/run_config.json` (`metricas_val_melhor_checkpoint`) e `outputs/baseline/baseline_results.json`.

| Método | Acurácia | Recall (recusada) | Especificidade | Precisão | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Majoritário | _ | _ | _ | _ | _ | _ |
| Histograma + LogReg | _ | _ | _ | _ | _ | _ |
| MobileNetV2 – Transfer | _ | _ | _ | _ | _ | _ |
| MobileNetV2 – Fine-Tuning | _ | _ | _ | _ | _ | _ |

Com poucas dezenas de imagens por conjunto, reportar também o IC95% (presente nos JSON) e evitar concluir a partir de diferenças pequenas.

## Análise de falhas

Listar as imagens erradas (`erro == True` em `predictions.csv`) e descrever padrões (material, iluminação, tipo de dano).

## Obstáculos e plano até a E3

_(preencher)_
