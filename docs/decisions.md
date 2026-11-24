# Decisões de projeto

Registro de decisões que afetam dados, rótulos ou experimentos. Cada decisão tem data e responsável.

## D1. Critério de rotulagem (base: README, seção 5, E1)

**RECUSADA** (dano que compromete a integridade): rasgo que expõe o conteúdo; perfuração; amassado que deforma a estrutura da embalagem.
**APROVADA**: embalagem íntegra, ou com desgaste leve (pequeno vinco, sujeira superficial).

> **GRUPO: confirmar/ajustar este texto antes de rotular e acrescentar exemplos.** Mudar o critério depois de rotular exige revisar os rótulos.

## D2. Divisão dos dados

- Agrupada por `id_embalagem`, estratificada por classe, seed 42, implementada em `src/prepare_data.py`
  (`StratifiedGroupKFold`: 7 dobras para teste, 6 sobre o restante para validação; proporção real ≈ 71/14/14, reportada em `data/split_report.md`).
- Todas as fotos de uma embalagem física (ângulos e versão intacta/danificada) ficam no mesmo conjunto.
- **Limitação conhecida:** a estratificação é só por classe. O tipo de embalagem (plástico/papel) é reportado por conjunto, mas
  não é garantido equilibrado nos conjuntos pequenos. Conferir a tabela do `split_report.md` antes de treinar.
- O teste só é usado em `src/evaluate.py --split test`, uma vez, após fixar a configuração.

## D3. Baselines

1. Classe majoritária do treino (esperado ≈ 50% de acurácia se as classes estiverem equilibradas).
2. Regressão logística sobre histograma de cor RGB. Serve de piso simples e de alarme: desempenho muito acima do acaso
   com esse modelo sugere diferença de iluminação/fundo entre as classes (viés de captura), não "visão de dano".

## D4. Treinamento

- Transfer Learning: só `classifier[1]` treina; `features` congelado e em modo `eval` (BatchNorm fixo). LR 1e-3.
- Fine-Tuning parcial: `features[14:]` + classificador; blocos anteriores congelados e em modo `eval`. LR 1e-4 (conforme plano da E1).
- Adam, CrossEntropyLoss, batch 16, até 30 épocas, early stopping (paciência 5) na perda de validação; mesma semente nas duas execuções.
