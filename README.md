# g6-em-inspecao

Inspeção automatizada de embalagens com MobileNetV2 e PyTorch, para a fábrica de bolachas Produtos Adrião.

Projeto de Tópicos Avançados em IA, 2026.2, Prof. Marcondes Ricarte.

Autores: Vitória Jamile e Eraldo Barros.

## Status do projeto

Entrega atual: **E1, definição do projeto**. Nenhuma imagem foi coletada e nenhum modelo foi treinado. Este README cobre problema, escopo, artigo base, modelo, estratégia de coleta e plano de avaliação, e será completado a cada entrega.

| Entrega | Conteúdo | Data | Situação |
|---|---|---|---|
| E1 | Definição do projeto | 13/11/2026 | Em revisão |
| E2 | Dados e primeira adaptação | 22/11/2026 | Pendente |
| E3 | Aplicação e resultados experimentais | 29/11/2026 | Pendente |
| E4 | Apresentação e entrega final | 07/12/2026 | Pendente |

## 1. Problema e contexto

O problema foi identificado a partir de um relato de um dos proprietários da Produtos Adrião, fábrica de bolachas que constitui o contexto de aplicação deste projeto. Segundo o proprietário, o processo de embalagem é realizado manualmente. Durante essa etapa, algumas embalagens podem apresentar falhas e, quando não são percebidas, acabam sendo encaixotadas e enviadas aos compradores.

Quando o comprador recebe um produto com a embalagem defeituosa, o item danificado precisa ser descartado e um novo produto é enviado para reposição. Essa ocorrência gera desperdício de mercadoria e de insumos, retrabalho, atrasos na logística e prejuízos diretos ao desempenho operacional da empresa. O relato aponta, portanto, uma oportunidade de realizar uma inspeção visual antes do encaixotamento, para identificar possíveis falhas antes do envio.

**Cliente e contexto de aplicação:** Produtos Adrião, fábrica de bolachas. **Beneficiários:** operadores responsáveis pela embalagem e inspeção, supervisores de qualidade e profissionais envolvidos no armazenamento e na expedição.

**Evidência inicial:** relato de um dos proprietários sobre o processo manual, a ocorrência de embalagens com falhas que seguem para os compradores e a necessidade de descarte do produto avariado com posterior envio de reposição. Essa evidência é inicial e qualitativa; não representa, por si só, uma medição da frequência de defeitos ou dos custos gerados. Caso estejam disponíveis, esses dados poderão ser levantados em etapas posteriores.

**Proposta:** inserir uma etapa de inspeção fotográfica antes do encaixotamento. Enquanto aguarda a análise, a embalagem pode ser identificada operacionalmente como **EM INSPEÇÃO**. No protótipo acadêmico, o usuário enviará uma fotografia à aplicação, que apresentará uma das duas classes previstas: **APROVADA** ou **RECUSADA**. A indicação servirá como apoio à avaliação humana; a decisão final continuará sob responsabilidade do operador.

**Limite da proposta:** o projeto implementa um protótipo de análise de fotografias e não contempla integração automática à linha de produção, acionamento de equipamentos ou inspeção em tempo real.

## 2. Objetivo e escopo

**Objetivo geral:** classificar fotografias de embalagens como íntegras ou danificadas usando a MobileNetV2, adaptada por transferência de aprendizado (*Transfer Learning*) e ajuste fino (*Fine-Tuning*).

**Fluxo previsto:** (1) a embalagem é produzida e fechada manualmente; (2) antes de ser encaixotada, entra na etapa **EM INSPEÇÃO** e é fotografada; (3) o usuário envia a fotografia à aplicação Streamlit; (4) a MobileNetV2 apresenta a classe prevista (**APROVADA** ou **RECUSADA**) e a confiança estimada pelo modelo (saída do *softmax*, não necessariamente calibrada); (5) o operador avalia a indicação e decide se a embalagem pode seguir para o encaixotamento ou se deve ser separada para verificação. No protótipo, o envio da fotografia é feito pelo usuário, sem integração com equipamentos da fábrica.

**Funcionalidades previstas:** envio de uma fotografia, visualização da imagem e apresentação da classe prevista e da confiança estimada pelo modelo.

**Classes de saída do modelo:** a classificação é binária. A expressão **EM INSPEÇÃO** representa apenas uma etapa temporária do fluxo operacional, antes da análise; ela não é uma terceira classe do modelo.

| Classe | Significado | Identificador no código |
|---|---|---|
| APROVADA | Embalagem íntegra | `aprovada` |
| RECUSADA | Embalagem danificada (classe positiva) | `recusada` |

**Fora do escopo:** identificar o tipo de defeito, localizar o dano na imagem, detectar múltiplos defeitos, integrar com linhas de produção e inspecionar em tempo real com câmeras industriais.

## 3. Artigo base

SHAHIN, M.; CHEN, F. F.; BOUZARY, H.; HOSSEINZADEH, A. Deploying Convolutional Neural Network to Reduce Waste in Production System. *Manufacturing Letters*, v. 35, p. 1187–1195, 2023. DOI: [10.1016/j.mfglet.2023.08.127](https://doi.org/10.1016/j.mfglet.2023.08.127).

**O que o artigo faz.** Treina um YOLOv7, que é um detector de objetos, para identificar embalagens danificadas e reduzir desperdício. Usa o conjunto público [Industrial Quality Control of Packages](https://www.kaggle.com/datasets/christianvorhemus/industrial-quality-control-of-packages), do Kaggle, com 400 imagens geradas no Blender (vistas de topo e lateral de 200 embalagens), anotadas com LabelMe, divididas 80/20 entre treinamento e teste, em 640 × 640, por 100 épocas no Google Colab. A Tabela 3 reporta precisão de 75%, recall de 69%, F1-score de 70% e acurácia de 67%. Esses números parecem ser médias entre as duas classes: a Tabela 1 indica recall de 94% para embalagens danificadas e 44% para íntegras. Como a Tabela 1 também tem valores que não fecham entre si (por exemplo, TN = 68 na classe "intact"), citamos só a Tabela 3 e não comparamos numericamente com ela.

**O que aproveitamos.** O problema e o contexto de redução de desperdício, e a ideia de usar redes convolucionais para separar embalagens íntegras de danificadas.

**O que fazemos diferente, e por quê.**

- Os autores concluem que são necessários conjuntos de dados reais e maiores. Por isso construímos uma base própria de fotografias reais.
- O artigo não diz se as duas vistas de uma mesma embalagem ficaram no mesmo conjunto na divisão 80/20. Por isso nossa divisão é agrupada por embalagem física.
- Em vez de detecção com YOLOv7 em imagens sintéticas, fazemos classificação binária com MobileNetV2 em fotografias reais. É uma escolha metodológica nossa, e os resultados não serão comparados com os do artigo.

## 4. Modelo

**Arquitetura:** MobileNetV2 (SANDLER et al., 2018), uma rede convolucional eficiente com blocos residuais invertidos e convoluções separáveis em profundidade. Ela é leve e funciona bem com bases pequenas quando partimos de pesos pré-treinados.

**Pesos pré-treinados:** implementação do torchvision com `MobileNet_V2_Weights.IMAGENET1K_V2`, indicados de forma explícita e não por `DEFAULT`, que pode mudar entre versões da biblioteca. Os pesos são disponibilizados pelo torchvision e foram pré-treinados para a tarefa original de classificação de imagens do ImageNet-1K (1.000 classes). A documentação da implementação e dos pesos está em [MobileNetV2 — torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html). A licença da implementação do torchvision é BSD-3-Clause; a licença e as condições de uso dos pesos e do conjunto ImageNet deverão ser verificadas e registradas antes da distribuição dos artefatos. A última camada, `Linear(1280, 1000)`, é trocada por `Linear(1280, 2)`, mantendo o dropout existente. As versões de PyTorch e torchvision serão fixadas no `requirements.txt`.

**Relação com o artigo e correspondência metodológica:** o artigo e este projeto usam redes convolucionais para separar embalagens íntegras de danificadas a partir de imagens. O artigo resolve a tarefa como detecção (YOLOv7, com caixas delimitadoras). Aqui mantemos só a decisão íntegra ou danificada por imagem, que é o que o operador precisa na etapa de inspeção, com um modelo mais leve, adequado a uma base pequena e a recursos computacionais limitados. Não reproduzimos a escala nem os resultados do artigo.

**Experimentos planejados**, com a mesma divisão de dados e as mesmas métricas:

1. **Transfer Learning:** extrator congelado (camadas de normalização em modo de avaliação) e treinamento apenas do novo classificador.
2. **Fine-Tuning parcial:** blocos finais descongelados, a partir de `features[14]` como configuração inicial, treinados com o classificador e com taxa de aprendizado menor. O ponto de corte pode ser ajustado na validação.

O objetivo é saber se ajustar parte das camadas pré-treinadas traz ganho sobre treinar só o classificador. Como a base é pequena, diferenças pequenas serão interpretadas com cautela.