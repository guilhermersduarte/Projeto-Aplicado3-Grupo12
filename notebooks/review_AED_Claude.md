---
## 12. O que Aprendemos: Resumo para o Sistema de Recomendação

Esta seção resume o que cada análise nos diz sobre como construir o sistema.

---

### 12.1 Limpeza necessária antes de usar os dados

- **Avaliações duplicadas** (mesmo usuário, mesmo curso): manter apenas a mais recente.
- **Texto ausente nas avaliações**: manter a linha para análise de notas, mas excluir da parte que usa texto.
- **Nomes de usuários**: remover o prefixo "By " para tratar como identificador.
- **Avaliações muito curtas** (menos de 5 palavras): pouco úteis para análise de texto — filtrar.

---

### 12.2 O que os dados nos dizem sobre recomendar por comportamento de usuários

- A grande maioria dos usuários tem **apenas uma avaliação**. Isso limita muito a capacidade de entender o gosto de cada um só pelo histórico.
- A matriz usuário × curso tem **densidade menor que 1%** — os dados são muito esparsos.
- **Conclusão**: A recomendação por comportamento de usuários funciona, mas precisa de regulação forte. Para usuários com poucas avaliações, deve ser complementada pela análise de conteúdo dos cursos.

---

### 12.3 O que os dados nos dizem sobre recomendar pelo conteúdo dos cursos

Características dos cursos confirmadas como úteis:
- **Palavras no nome do curso** (ex: "machine learning", "data", "business")
- **Nota média e variação das notas** por curso
- **Reputação da instituição** (nota média ponderada pelo volume)
- **Texto das avaliações** com mais de 20 palavras (o restante tem pouco conteúdo)

---

### 12.4 Como lidar com usuários e cursos novos (sem histórico)

- Cursos com poucas avaliações: **sem dados suficientes para recomendação por comportamento**. Usar apenas análise de conteúdo.
- Usuário novo sem histórico: **recomendar os "Destaques"** — cursos populares e bem avaliados — filtrados por área de interesse declarada.

---

### 12.5 Como dividir os dados para treino e teste

- **Treino**: avaliações feitas antes de 2020.
- **Teste**: avaliações de 2020 em diante.
- Justificativa: o período da pandemia gerou um pico de uso que representa um cenário real de crescimento acelerado — testar nele é mais realista.

---

### 12.6 Como medir se o sistema está funcionando bem

- **Precisão@K**: dos K cursos sugeridos, quantos o usuário realmente gostaria?
- **NDCG@K**: os melhores cursos aparecem no topo da lista?
- **Cobertura**: o sistema está recomendando uma variedade de cursos, ou sempre os mesmos populares?

> **Atenção**: como poucos cursos concentram a maioria das avaliações, há risco de o sistema sempre recomendar os mesmos. A métrica de cobertura ajuda a detectar isso.