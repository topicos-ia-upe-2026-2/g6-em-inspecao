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