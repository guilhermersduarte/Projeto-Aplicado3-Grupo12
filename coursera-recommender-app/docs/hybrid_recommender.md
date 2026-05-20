# Documentação: Migração para o HybridRecommender

## Por que mudamos?

O arquivo `src/recommender.py` foi substituído por `src/hybrid_recommender.py`.

A versão anterior usava TF-IDF puro em todos os casos e nunca chegava a usar o modelo SVD treinado (os IDs de usuário do dataset original são incompatíveis com os IDs do Supabase). A nova versão resolve isso com uma técnica chamada **SVD fold-in**: em vez de precisar que o usuário esteja no dataset de treino, o modelo calcula um vetor de usuário na hora, a partir das avaliações que ele deu no app.

| Aspecto | Antigo (`recommender.py`) | Novo (`hybrid_recommender.py`) |
|---|---|---|
| Modelos | TF-IDF apenas | TF-IDF + SVD fold-in |
| SVD | Carregado, nunca usado | Ativado após 5 avaliações |
| Estratégia | Sempre cold start manual | 3 estágios progressivos |
| Busca por texto | Não existia | `search_by_text()` |
| Estado por usuário | Sim (stateful) | Não — ratings passados por parâmetro |

---

## Arquitetura: os 3 estágios

A classe `HybridRecommender` (em [src/hybrid_recommender.py](../src/hybrid_recommender.py)) é **stateless**: ela não guarda dados do usuário. A cada chamada, você passa o histórico de avaliações como dicionário `{course_id: nota}`, e ela decide qual estágio aplicar.

```
cold_start       →  0 avaliações         →  busca por texto livre (TF-IDF)
tfidf_profile    →  1 a 4 avaliações     →  perfil TF-IDF (média ponderada dos cursos avaliados)
svd_fold_in      →  ≥ 5 avaliações SVD   →  SVD fold-in (filtragem colaborativa)
```

O estágio é determinado pelo método `user_status()`, que verifica quantas das avaliações do usuário correspondem a cursos que o modelo SVD conhece (cursos do dataset de treino).

### Estágio 1 — cold_start

Usuário sem nenhuma avaliação. O app exibe um formulário de texto livre ("Descreva seus interesses") e chama `search_by_text(query)`, que reconstrói o vectorizer TF-IDF a partir dos parâmetros salvos em `tfidf_params.json` e retorna os cursos mais similares à query.

### Estágio 2 — tfidf_profile

Usuário com 1 a 4 avaliações em cursos conhecidos pelo SVD. O método `_build_tfidf_profile()` calcula uma **média ponderada** dos vetores TF-IDF dos cursos avaliados com nota ≥ `rating_floor` (padrão: 3.0). O resultado é um vetor que representa o "gosto" do usuário no espaço TF-IDF, e a similaridade de cosseno seleciona os mais próximos.

### Estágio 3 — svd_fold_in

Usuário com ≥ 5 avaliações em cursos que o SVD conhece. O método `_fold_in_svd()` usa mínimos quadrados regularizados (regularização L2, `reg=0.02`) para computar o vetor latente `p_u` do usuário, sem retreinar o modelo. A predição de nota para cada curso é:

```
nota_predita = μ + b_u + b_i + p_u · q_i
```

onde `μ` é a média global, `b_u` e `b_i` são os biases de usuário e item, e `q_i` é o vetor latente do curso.

---

## Parâmetros configuráveis

Definidos no construtor da classe, em [app.py](../app.py) na função `carregar_recomendador()`:

| Parâmetro | Padrão | Significado |
|---|---|---|
| `svd_fold_in_threshold` | `5` | Mínimo de avaliações em cursos SVD para ativar o fold-in |
| `rating_floor` | `3.0` | Nota mínima para incluir um curso no perfil TF-IDF |
| `reg` | `0.02` | Regularização L2 no cálculo do fold-in |
| `svd_pool_ratio` | `0.7` | Proporção de resultados vindos de cursos do SVD vs. exploração |

---

## API pública

Todos os métodos relevantes para o `app.py`:

```python
# Carrega os modelos (chamado uma vez via @st.cache_resource)
rec.load(svd_path, tfidf_matrix_path, tfidf_meta_path, tfidf_params_path)

# Retorna o estágio atual e progresso do usuário
status = rec.user_status(ratings)
# → {"stage": "tfidf_profile", "total_ratings": 2, "svd_known_ratings": 2,
#    "ratings_to_fold_in": 3, "svd_fold_in_active": False}

# Gera recomendações para usuários com histórico (estágios 2 e 3)
recs = rec.recommend(ratings, top_n=10, exclude=cursos_ja_avaliados)
# → [("machine-learning", 0.87), ("deep-learning", 0.81), ...]

# Busca por texto livre (estágio 1 — cold start)
recs = rec.search_by_text("machine learning python", top_n=5)
# → [("python-for-everybody", 0.74), ...]

# Recomenda cursos similares a uma lista de cursos-semente
recs = rec.suggest_from_seed_courses(["machine-learning"], top_n=10)
# → [("deep-learning", 0.91), ...]  (70% cursos SVD + 30% exploração)
```

`ratings` é sempre um `dict[str, float]` no formato `{course_id: nota}`, por exemplo:
```python
{"machine-learning": 5.0, "python-for-everybody": 4.0}
```

---

## Como o app.py usa o recomendador

O recomendador é carregado uma vez por instância do servidor (uma instância compartilhada por todos os usuários):

```python
@st.cache_resource
def carregar_recomendador() -> HybridRecommender:
    rec = HybridRecommender(svd_fold_in_threshold=5)
    rec.load(
        svd_path="data/models/svd_model.pkl",
        tfidf_matrix_path="data/tfidf_matrix.npz",
        tfidf_meta_path="data/tfidf_meta.parquet",
        tfidf_params_path="data/tfidf_params.json",
    )
    return rec
```

A cada sessão, o histórico é lido do Supabase e convertido para o formato de ratings:

```python
historico = buscar_historico(user_id)           # [{curso_id, nota}, ...]
ratings = {h["curso_id"]: float(h["nota"]) for h in historico}
status = rec.user_status(ratings)
```

O app então toma um de dois caminhos:

- **cold_start** → exibe formulário de texto e chama `rec.search_by_text(preferencias)`
- **tfidf_profile / svd_fold_in** → chama `rec.recommend(ratings, exclude=cursos_avaliados)` diretamente

---

## Fluxo completo

```
Usuário acessa o app
        │
        ▼
Autenticação OTP via Supabase (email → código numérico)
        │
        ▼
buscar_historico(user_id) → ratings = {course_id: nota}
        │
        ▼
rec.user_status(ratings)
        │
        ├── cold_start  ──► Formulário de texto livre
        │                         │
        │                         ▼
        │                   rec.search_by_text(preferencias) → Top-5
        │
        ├── tfidf_profile ─► rec.recommend(ratings) → perfil TF-IDF → Top-5
        │
        └── svd_fold_in ──► rec.recommend(ratings) → fold-in SVD → Top-5
                                                        │
                                                        └── 70% cursos SVD
                                                            30% exploração TF-IDF
        │
        ▼
Usuário avalia cada curso (slider 1–5)
        │
        ▼
salvar_feedback() + salvar_interacao() → Supabase
        │
        ▼
Na próxima sessão: histórico maior → pode avançar de estágio
```

---

## Decisões técnicas importantes

**Por que stateless?**
A instância é compartilhada entre todos os usuários via `@st.cache_resource`. Se o estado do usuário ficasse na classe, um usuário sobrescreveria os dados de outro. Passar `ratings` como parâmetro resolve isso de forma simples.

**Por que fold-in em vez de retreinar?**
Retreinar o SVD a cada novo usuário seria lento e inviável. O fold-in projeta o usuário novo no espaço latente já treinado, sem alterar os vetores dos cursos (`q_i`). É uma aproximação, mas funciona bem para recomendação.

**Por que `rating_floor=3.0` no perfil TF-IDF?**
Incluir cursos com notas baixas no perfil poderia direcionar as recomendações para temas que o usuário não gostou. O piso de 3.0 filtra avaliações negativas ou neutras.

**Por que `svd_pool_ratio=0.7` no `suggest_from_seed_courses`?**
Garante que 70% dos resultados venham de cursos que o SVD conhece (útil para acelerar a ativação do fold-in) e 30% de exploração no catálogo completo.
