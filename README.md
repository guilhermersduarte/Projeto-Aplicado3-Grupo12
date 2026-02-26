# 🎓 Sistema de Recomendação de Cursos Educacionais

### *Personalizando o aprendizado com Inteligência Artificial*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow.svg)]()


---

## 📋 Descrição do Projeto

Este projeto foi desenvolvido como parte da disciplina **Projeto Aplicado III** do curso de Ciência de Dados da Universidade Presbiteriana Mackenzie. O objetivo é construir um sistema inteligente de recomendação de cursos educacionais que auxilie estudantes a encontrarem conteúdos alinhados aos seus perfis, necessidades e objetivos de carreira.

### 🎯 Problema Abordado

Com o crescimento exponencial das plataformas de ensino online (MOOCs), estudantes enfrentam um desafio crescente: a **sobrecarga de informação**. Com milhares de cursos disponíveis, torna-se difícil identificar quais opções são mais adequadas ao nível de conhecimento, interesses e objetivos profissionais de cada pessoa.

### 💡 Solução Proposta

Desenvolveremos um sistema de recomendação híbrido que combina múltiplas abordagens de Machine Learning:

| Componente | Tecnologia | Descrição |
|------------|------------|-----------|
| **Filtragem Colaborativa** | SVD, NCF | Identifica padrões de preferência entre usuários similares |
| **Filtragem Baseada em Conteúdo** | TF-IDF, Embeddings | Analisa características dos cursos para recomendações |
| **Deep Learning** | PyTorch | Modelagem neural de embeddings de usuários e itens |
| **Cold Start Handling** | Híbrido | Estratégias para novos usuários sem histórico |


---

### 📊 Base de Dados

- **Fonte**: [Kaggle - Course Reviews on Coursera](https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera)
- **Volume**: ~1,45 milhão de avaliações reais
- **Cobertura**: Mais de 3.500 cursos e usuários da plataforma Coursera

---

## 🗂️ Estrutura do Repositório

```
.
│
├── 📁 docs/
│
├── 📁 data/
│
├── 📁 notebooks/
│
├── 📁 src/
│
├── LICENSE
│
└── README.md
```

