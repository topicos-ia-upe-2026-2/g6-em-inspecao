# Relatório de progresso – E2 (Dados e primeira adaptação)

Grupo G6 · Vitória Jamile e Eraldo Barros · Entrega: 22/11/2026 (alinhamento: 23/11/2026)

**Situação geral:** o pipeline de dados, baseline e treino está implementado em `src/`, mas a rotulagem das imagens não foi concluída. Por isso a divisão final, o baseline medido e o treino **não foram executados** nesta entrega. Nenhum resultado de modelo é apresentado.

## 1. Dados

| Item | Situação |
|---|---|
| Autorização da empresa para uso/compartilhamento das imagens | PREENCHER (sim/não e como foi registrada) |
| Imagens recebidas | 200: 22 fotos originais e 178 quadros extraídos de 7 vídeos |
| Imagens rotuladas (aprovada / recusada) | 0 / 0. As 29 fontes estão como `duvidosa` em `data/fontes.csv` |
| Embalagens/fontes distintas | 29 (cada vídeo e cada foto é um `id_embalagem`, para evitar vazamento entre quadros quase idênticos) |
| Dispositivos e resolução | Celular; arquivos recebidos via WhatsApp, resoluções variadas (de 474×850 a 1200×1600) |
| Local do conjunto completo (link externo) | PREENCHER (link) |
| `data/metadata.csv` validado por `src/prepare_data.py` | Não gerado ainda: depende da rotulagem |

**Por que a rotulagem está pendente:**
- O critério da E1 considera dano na **embalagem**. As imagens recebidas mostram sobretudo bolachas quebradas ou furadas dentro de pacotes, e a avaliação do dano de embalagem ainda é incerta.
- Não há fotos de embalagens íntegras para a classe `aprovada`.
- Divisão prevista: agrupada por fonte, com no máximo 8 quadros por vídeo (decisões D2 e D5 em `docs/decisions.md`).

## 2. Baseline

Não executado: exige rótulos nas duas classes. O código está em `src/baseline.py` (classe majoritária e histograma de cor com regressão logística).

## 3. Primeira execução de fine-tuning

Não executada. `src/train.py` implementa Transfer Learning e Fine-Tuning parcial (D4), mas não foi rodado com dados reais.

## 4. Resultados, limitações e obstáculos

- Sem resultados de modelo nesta entrega.
- Obstáculos: ausência de exemplos íntegros; incerteza no critério de rotulagem; poucas fontes independentes (≈29), o que tornará validação e teste frágeis e os intervalos de confiança largos.
- Os scripts de dados e métricas foram verificados apenas com dados sintéticos de teste, não com resultado de modelo.

## 5. Plano até a E3 (29/11)

1. Fotografar 15–20 pacotes íntegros e alguns danificados, com o mesmo fundo, luz e celular.
2. Definir com o professor o critério de rotulagem (só embalagem ou também defeito visível do produto) e rotular as fontes.
3. Rodar `prepare_data.py`, `baseline.py` e `train.py` nos dois modos, e preencher `docs/experiments.md` com os valores medidos.
4. Aplicação Streamlit, avaliação final no teste (uma vez), análise de falhas e teste de uso com um representante do público-alvo.