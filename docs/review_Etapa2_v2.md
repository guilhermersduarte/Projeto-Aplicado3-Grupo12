# Revisão — Etapa 2 v2

## 1. Coerência entre documento e notebooks

### O que o documento diz vs. o que os notebooks fazem

| Afirmação no documento | Notebook correspondente | Status |
|------------------------|------------------------|--------|
| Densidade de 0,84% na matriz de interações | NB01 calcula densidade; NB02 recalcula após filtros USS/ISS | **Verificar**: 0,84% é a densidade bruta (antes de filtros). Após USS/ISS a densidade sobe para ~5%. O documento não distingue os dois cenários. |
| 95,2% dos usuários com nota média ≥ 4,0 | NB01 Seção 7.1 confirma 95,2% | OK |
| De 622 cursos para 233 após USS/ISS (ref. Suryani) | NB02 Seção 5 aplica USS/ISS e chega a ~233 cursos | OK — mas o catálogo original tem 623 cursos (não 622). O documento diz "622" seguindo Suryani; o notebook usa 623. Pequena discrepância. |
| 1,45 milhão de registros para ~39 mil após filtros | NB02 confirma redução drástica | OK |
| Dataset = "100K Coursera Course Reviews" (Objetivos Específicos) | NB01 carrega 1.454.711 reviews | **Inconsistência**: o dataset real tem 1,45M reviews, não 100K. O nome "100K" é o título original do Kaggle, mas pode confundir o leitor. Esclarecer no texto. |
| Métricas: RMSE, MAE e Precisão@K | NB03 calcula exatamente essas três métricas | OK |
| SVD e/ou KNN como técnicas principais | NB03 implementa KNN Item-Based, KNN User-Based e SVD | OK |
| TF-IDF como componente de cold start | NB02 Seção 8 gera matriz TF-IDF (Ramo B) | OK — mas nenhum notebook usa TF-IDF para recomendação ainda. O pipeline produz a matriz, mas a aplicação fica para próxima etapa. |
| Split temporal: pré-2020 = treino, 2020+ = teste | NB02 Seção 7 faz exatamente isso | OK |
| Ref. Suryani K=150: RMSE=0,7882 MAE=0,4791 | NB03 reporta esses valores como baseline | OK |
| Abordagem híbrida com switching | Não implementada em nenhum notebook | **Pendente**: o documento descreve switching como estratégia, mas os notebooks ainda não têm esse mecanismo. Coerente com o plano (piloto futuro), mas vale deixar explícito no texto que é planejado, não implementado. |

### Números do NB03 que o documento não menciona (considerar incluir)

| Métrica | KNN Item | KNN User | SVD |
|---------|----------|----------|-----|
| RMSE | 0,8369 | 0,8907 | 0,7158 |
| MAE | 0,5058 | 0,5199 | 0,4596 |
| P@5 | 0,3857 | 0,3840 | 0,3856 |
| P@10 | 0,2129 | 0,2127 | 0,2129 |

O SVD supera tanto o KNN Item quanto o KNN User, e bate os baselines de Suryani. Esses resultados são o proof of concept da Etapa 2 — o documento deveria apresentá-los.

---

## 2. Seção ausente: Apresentação dos Resultados da Etapa 2

O template do Módulo 2 exige uma seção de **apresentação dos resultados** (vale 0,5 pontos). O documento atual termina após o Referencial Teórico e as Referências. Falta:

- Resumo do pipeline implementado (NB02)
- Resultados do proof of concept (NB03): tabela comparativa dos modelos, gráficos RMSE/MAE
- Menção ao estado do TF-IDF (Ramo B pronto, aplicação na próxima etapa)
- Próximos passos previstos

Sem essa seção, o documento perde 0,5 pontos diretos no critério de avaliação.

---

## 3. Revisão de escrita (stop-slop + padrões do projeto)

### Padrões detectados

**Ritmo monótono em vários parágrafos.** A seção de Referencial Teórico tem parágrafos com estrutura repetitiva: frase de contexto → explicação → consequência. Quase todos têm entre 4 e 6 linhas. Variar o comprimento quebraria a monotonia.

**Estruturas binárias formulaicas.** Aparecem pelo menos três vezes:

- "A relevância deste projeto se sustenta em dois eixos. O primeiro é acadêmico (...). O segundo eixo é social (...)" — funciona aqui por ser uma divisão real, mas a construção "primeiro/segundo" é usada de novo logo abaixo com as ODS.
- "O primeiro é a ODS 4 (...). O segundo é a ODS 8 (...)" — repetição da mesma estrutura dois parágrafos depois.
- "Nos métodos baseados em memória (...). Nos métodos baseados em modelo (...)" — aqui é adequado (são de fato duas famílias).

**Sugestão**: variar a forma de apresentar pares. Nem todo par precisa de "primeiro/segundo".

**Frases que anunciam em vez de entregar.**

- "Esta seção apresenta os fundamentos teóricos que sustentam as decisões técnicas do projeto." — Frase de abertura do Referencial Teórico que poderia ser cortada. O leitor sabe o que a seção faz pelo título.
- "São descritos os conceitos de sistemas de recomendação, as técnicas de filtragem colaborativa adotadas, a abordagem híbrida (...)" — Lista o conteúdo antes de entregá-lo. Cortar e ir direto ao primeiro parágrafo substantivo.

**Intensificadores sem carga.**

- "profundamente" em "transformou profundamente" (1.1) — "transformou" já carrega o peso.
- "fortemente" em "dependem fortemente" (1.1) — "dependem" basta.
- "especialmente crítica" (1.1) — "crítica" já é forte.
- "diretamente" aparece 5 vezes no documento. Algumas podem ser cortadas sem perda.

**Jargão evitável.**

- "pipeline completo de ciência de dados" (1.3) — OK no contexto técnico, mas "pipeline" aparece 4 vezes no documento inteiro. Alternar com "fluxo" ou "sequência" em alguns casos.

### Problemas de consistência terminológica

- O documento usa "aprendizes" e "estudantes" de forma intercambiável. Padronizar um termo.
- "cold start" aparece sem tradução na primeira ocorrência. Depois é usado livremente. OK para texto técnico de graduação, mas vale uma nota na primeira menção: "(cold start — partida a frio)".
- "Coursera/Kaggle" aparece uma vez. Em outro trecho é "plataforma Coursera". O dataset é do Kaggle, os cursos são do Coursera. Separar claramente: "dataset disponível no Kaggle, originário do Coursera".

### Erros factuais menores

- "622 cursos" (parágrafo sobre USS/ISS) — o catálogo tem 623. O 622 vem de Suryani. Se for citação do artigo, deixar claro. Se for dado do projeto, corrigir para 623.
- "100K Coursera Course Reviews" (Objetivos Específicos) — o dataset tem 1,45M reviews. O "100K" é parte do nome original no Kaggle, mas não é o volume real. Reformular para evitar confusão.

---

## 4. Resumo de ações sugeridas

| Prioridade | Ação |
|------------|------|
| **Alta** | Adicionar seção de resultados da Etapa 2 (0,5 pts no template) |
| **Alta** | Incluir tabela comparativa dos modelos (NB03) na seção de resultados |
| **Média** | Corrigir "622 cursos" → esclarecer se é dado do projeto (623) ou citação de Suryani |
| **Média** | Esclarecer "100K" no nome do dataset vs. 1,45M reviews reais |
| **Média** | Cortar parágrafo de abertura do Referencial Teórico (anuncia sem entregar) |
| **Baixa** | Reduzir intensificadores ("profundamente", "fortemente", "especialmente", "diretamente") |
| **Baixa** | Variar estrutura "primeiro/segundo" — não usar duas vezes em sequência |
| **Baixa** | Padronizar "aprendizes" vs. "estudantes" |
| **Baixa** | Adicionar "(cold start — partida a frio)" na primeira menção |
