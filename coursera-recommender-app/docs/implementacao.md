# Implementação da Interface e Sistema de Recomendação

## Estrutura do Projeto

O sistema foi desenvolvido como uma aplicação web utilizando o framework Streamlit, organizado nos seguintes módulos:

- **`app.py`** — ponto de entrada da aplicação, responsável pela interface e orquestração do fluxo
- **`src/database.py`** — camada de acesso ao banco de dados Supabase
- **`src/recommender.py`** — lógica de carregamento dos modelos e geração de recomendações

---

## Autenticação

A autenticação foi implementada via Supabase Auth utilizando o método OTP (*One-Time Password*) por email. O fluxo funciona da seguinte forma: o usuário informa seu email, recebe um código numérico de 8 dígitos e o insere na interface para validação. Optou-se por OTP em vez de magic link porque o Streamlit não consegue capturar tokens de URL (que chegam como fragmentos de hash `#`), tornando o magic link incompatível com a arquitetura do framework.

O estado de autenticação é verificado a cada carregamento via `supabase.auth.get_session()`. O identificador único do usuário (`user_id`) gerado pelo Supabase é utilizado em todas as operações subsequentes de leitura e escrita no banco.

---

## Carregamento dos Modelos

Os artefatos de modelo são carregados uma única vez por sessão utilizando o decorator `@st.cache_resource` do Streamlit, evitando recarregamentos desnecessários a cada interação do usuário. Os seguintes artefatos são carregados:

- `data/models/svd_model.pkl` — modelo SVD treinado com Surprise
- `data/tfidf_matrix.npz` — matriz TF-IDF esparsa (623 cursos × 5.000 features)
- `data/tfidf_meta.parquet` — metadados dos cursos (identificadores)
- `data/tfidf_params.json` — vocabulário e pesos IDF do vectorizer

O vectorizer TF-IDF é reconstruído em tempo de execução a partir dos parâmetros salvos, com restauração explícita dos atributos internos `vocabulary_` e `_tfidf.idf_`, necessários para que o scikit-learn o reconheça como um modelo já ajustado (*fitted*).

---

## Lógica de Recomendação

A arquitetura implementa um mecanismo de *switching* baseado no histórico de avaliações do usuário, com dois caminhos:

**Cold start (usuário sem histórico ou com menos de 3 avaliações):**
O usuário preenche um campo de texto descrevendo seus interesses (ex.: *"machine learning, python, análise de dados"*). Essa entrada é transformada em vetor TF-IDF utilizando o mesmo vocabulário do corpus de treinamento e comparada com os vetores dos 623 cursos via similaridade de cosseno. Os 5 cursos com maior similaridade são retornados como recomendação.

**Usuário com histórico (3 ou mais avaliações registradas):**
Em vez de utilizar o modelo SVD — que foi treinado com identificadores de usuários do dataset original do Coursera e não reconhece os novos usuários do piloto — optou-se por uma abordagem de perfil baseada em conteúdo. O sistema recupera os cursos avaliados com nota ≥ 4 pelo usuário, concatena seus nomes como query e executa a mesma busca TF-IDF. Cursos já avaliados são excluídos dos resultados. Caso nenhum curso tenha atingido nota ≥ 4, todos os cursos avaliados são utilizados na query. Essa decisão garante personalização real sem depender de um modelo colaborativo treinado em dados externos.

---

## Coleta de Feedback

Após exibição das recomendações, o usuário avalia cada curso em uma escala de 1 a 5 (sendo 1 = nenhum interesse e 5 = muito interesse) através de sliders individuais. Ao submeter o formulário, duas operações são realizadas no Supabase:

- **Tabela `interacoes`**: registra quais cursos foram exibidos, em qual posição e por qual modelo
- **Tabela `feedback`**: registra a nota atribuída pelo usuário a cada curso

Ambas as tabelas têm Row Level Security (RLS) ativado, garantindo que cada usuário acesse apenas seus próprios dados.

---

## Decisões Técnicas Relevantes

- **Python 3.11** foi adotado como versão de ambiente, pois `scikit-surprise` não possui wheels pré-compiladas para Python 3.12+ no Windows e exige compilador C++. O Streamlit Cloud (Linux) compila sem problemas.
- **NumPy < 2.0** foi fixado no `requirements.txt` por incompatibilidade do `scikit-surprise` com a API de tipos do NumPy 2.x (`np.int_t` removido).
- O modelo SVD foi mantido no repositório e carregado na inicialização, mas não é utilizado no fluxo de recomendação do piloto pela razão descrita acima. Sua utilização efetiva dependeria de retreinamento com os identificadores dos usuários reais do sistema.
