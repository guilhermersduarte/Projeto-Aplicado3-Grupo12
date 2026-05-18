"""Recomendador baseado em conteudo via TF-IDF + similaridade de cosseno.

Usa a matriz `data/tfidf_matrix.npz` (623 cursos x 5000 termos) construida
no notebook 02. O perfil de cada usuario e a media ponderada pelas avaliacoes
dos cursos que ele ja avaliou no treino.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, rating_floor: float = 3.0):
        # Avaliacoes >= rating_floor contam como sinal positivo no perfil.
        self.rating_floor = rating_floor
        self.tfidf = None                 # scipy sparse (n_courses, n_terms)
        self.course_ids: np.ndarray | None = None
        self.cid_to_row: dict[str, int] = {}
        self.user_profiles: dict | None = None  # uid -> vetor denso (n_terms,)

    def load(self, matrix_path: str | Path, meta_path: str | Path) -> "ContentRecommender":
        self.tfidf = sp.load_npz(matrix_path).tocsr()
        meta = pd.read_parquet(meta_path)
        self.course_ids = meta["course_id"].to_numpy()
        self.cid_to_row = {cid: i for i, cid in enumerate(self.course_ids)}
        return self

    # ----- Similaridade item-item -----
    def similar_courses(self, course_id: str, top_n: int = 10) -> list[tuple[str, float]]:
        if course_id not in self.cid_to_row:
            return []
        row = self.cid_to_row[course_id]
        sims = cosine_similarity(self.tfidf[row], self.tfidf).ravel()
        sims[row] = -1.0  # exclui o proprio
        idx = np.argpartition(-sims, top_n)[:top_n]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.course_ids[i], float(sims[i])) for i in idx]

    # ----- Perfil de usuario (media ponderada dos vetores TF-IDF dos cursos avaliados) -----
    def build_user_profiles(self, train_ratings: pd.DataFrame) -> None:
        """train_ratings: DataFrame com colunas reviewers, course_id, rating."""
        df = train_ratings[train_ratings["rating"] >= self.rating_floor].copy()
        df = df[df["course_id"].isin(self.cid_to_row)]
        df["row"] = df["course_id"].map(self.cid_to_row)

        profiles = {}
        for uid, grp in df.groupby("reviewers", observed=True):
            rows = grp["row"].to_numpy()
            weights = grp["rating"].to_numpy(dtype=np.float32)
            vecs = self.tfidf[rows].toarray().astype(np.float32)  # (n_i, n_terms)
            prof = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
            profiles[uid] = prof
        self.user_profiles = profiles

    def recommend_for_user(self, user_id: str, top_n: int = 10,
                           exclude: set | None = None) -> list[tuple[str, float]]:
        if self.user_profiles is None or user_id not in self.user_profiles:
            return []
        prof = self.user_profiles[user_id].reshape(1, -1)
        sims = cosine_similarity(prof, self.tfidf).ravel()
        if exclude:
            for cid in exclude:
                if cid in self.cid_to_row:
                    sims[self.cid_to_row[cid]] = -1.0
        idx = np.argpartition(-sims, top_n)[:top_n]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.course_ids[i], float(sims[i])) for i in idx]

    def predict_score(self, user_id: str, course_id: str,
                      fallback: float = 3.5) -> float:
        """Devolve um score [0,1] mapeado para rating [1,5].

        Se o usuario nao tem perfil (cold start puro) ou o curso nao esta
        no vocabulario TF-IDF, retorna `fallback` (media neutra).
        """
        if (self.user_profiles is None or user_id not in self.user_profiles
                or course_id not in self.cid_to_row):
            return fallback
        prof = self.user_profiles[user_id]
        row = self.cid_to_row[course_id]
        vec = self.tfidf[row].toarray().ravel()
        denom = (np.linalg.norm(prof) * np.linalg.norm(vec))
        if denom == 0:
            return fallback
        sim = float(np.dot(prof, vec) / denom)  # [0, 1] tipicamente
        # Mapeia similaridade -> rating. Ancora: sim=0 -> 3.0, sim=1 -> 5.0.
        return float(np.clip(3.0 + 2.0 * sim, 1.0, 5.0))

    def predict_batch(self, test_triples: Iterable[tuple]) -> list[tuple]:
        """Retorna lista no formato Surprise: (uid, iid, r_ui, est, {})."""
        out = []
        for uid, iid, true_r in test_triples:
            est = self.predict_score(uid, iid)
            out.append((uid, iid, float(true_r), est, {}))
        return out

    def cleanup(self) -> None:
        self.tfidf = None
        self.user_profiles = None
        gc.collect()
