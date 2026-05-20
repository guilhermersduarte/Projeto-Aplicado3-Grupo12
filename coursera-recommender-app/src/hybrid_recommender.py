from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    """
    Recomendador híbrido TF-IDF + SVD fold-in.

    Stateless por usuário: ratings são sempre passados como parâmetro.
    Deve ser carregado via @st.cache_resource — uma instância por servidor.

    Estágios:
      cold_start    — sem avaliações → usar suggest_from_seed_courses()
      tfidf_profile — tem avaliações, mas poucas em cursos SVD → perfil TF-IDF
      svd_fold_in   — >= svd_fold_in_threshold avaliações em cursos SVD → fold-in
    """

    def __init__(
        self,
        svd_fold_in_threshold: int = 5,
        reg: float = 0.02,
        rating_floor: float = 3.0,
        svd_pool_ratio: float = 0.7,
    ):
        self.svd_fold_in_threshold = svd_fold_in_threshold
        self.reg = reg
        self.rating_floor = rating_floor
        self.svd_pool_ratio = svd_pool_ratio

        # Preenchidos por load()
        self.svd = None
        self.tfidf_matrix: sp.csr_matrix | None = None
        self.tfidf_params: dict | None = None
        self.course_ids: list[str] = []
        self.cid_to_tfidf_row: dict[str, int] = {}

        # Preenchidos por _build_translation()
        self.cid_to_svd_inner: dict[str, int] = {}
        self.svd_inner_to_cid: dict[int, str] = {}
        self.svd_course_ids: set[str] = set()
        self.svd_rows_in_tfidf: list[int] = []

    # ------------------------------------------------------------------
    # Carregamento
    # ------------------------------------------------------------------

    def load(
        self,
        svd_path: str | Path,
        tfidf_matrix_path: str | Path,
        tfidf_meta_path: str | Path,
        tfidf_params_path: str | Path | None = None,
    ) -> "HybridRecommender":
        import json

        try:
            from surprise import dump
            _, svd = dump.load(str(svd_path))
        except Exception:
            import pickle
            with open(svd_path, "rb") as f:
                svd = pickle.load(f)
        self.svd = svd

        self.tfidf_matrix = sp.load_npz(str(tfidf_matrix_path))
        meta = pd.read_parquet(str(tfidf_meta_path))
        self.course_ids = meta["course_id"].tolist()
        self.cid_to_tfidf_row = {cid: i for i, cid in enumerate(self.course_ids)}

        if tfidf_params_path is not None:
            with open(tfidf_params_path) as f:
                self.tfidf_params = json.load(f)

        self._build_translation()
        return self

    def _build_translation(self) -> None:
        trainset = self.svd.trainset
        for inner_id in range(trainset.n_items):
            raw_id = trainset.to_raw_iid(inner_id)
            self.cid_to_svd_inner[raw_id] = inner_id
            self.svd_inner_to_cid[inner_id] = raw_id

        self.svd_course_ids = set(self.cid_to_svd_inner)
        self.svd_rows_in_tfidf = [
            self.cid_to_tfidf_row[cid]
            for cid in self.svd_course_ids
            if cid in self.cid_to_tfidf_row
        ]

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def suggest_from_seed_courses(
        self,
        seed_course_ids: list[str],
        top_n: int = 10,
        exclude: set | None = None,
    ) -> list[tuple[str, float]]:
        exclude_set = (set(exclude) if exclude else set()) | set(seed_course_ids)

        seed_rows = [
            self.cid_to_tfidf_row[cid]
            for cid in seed_course_ids
            if cid in self.cid_to_tfidf_row
        ]
        if not seed_rows:
            return []

        query_vec = np.asarray(self.tfidf_matrix[seed_rows].mean(axis=0))
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        excluded_rows = {
            self.cid_to_tfidf_row[cid]
            for cid in exclude_set
            if cid in self.cid_to_tfidf_row
        }

        svd_row_set = set(self.svd_rows_in_tfidf)

        svd_pool = sorted(
            [
                (self.course_ids[i], float(sims[i]))
                for i in self.svd_rows_in_tfidf
                if i not in excluded_rows
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        explore_pool = sorted(
            [
                (self.course_ids[i], float(sims[i]))
                for i in range(len(self.course_ids))
                if i not in svd_row_set and i not in excluded_rows
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        n_svd = int(top_n * self.svd_pool_ratio)
        n_explore = top_n - n_svd
        return svd_pool[:n_svd] + explore_pool[:n_explore]

    def recommend(
        self,
        ratings: dict[str, float],
        top_n: int = 10,
        exclude: set | None = None,
    ) -> list[tuple[str, float]]:
        exclude_set = (set(exclude) if exclude else set()) | set(ratings.keys())

        svd_params = self._fold_in_svd(ratings)
        if svd_params is not None:
            p_u, b_u = svd_params
            return self._recommend_svd(p_u, b_u, top_n, exclude_set)

        profile = self._build_tfidf_profile(ratings)
        if profile is not None:
            return self._recommend_tfidf_profile(profile, top_n, exclude_set)

        return []

    def user_status(self, ratings: dict[str, float]) -> dict:
        n_svd = sum(1 for cid in ratings if cid in self.cid_to_svd_inner)
        total = len(ratings)

        if n_svd >= self.svd_fold_in_threshold:
            stage = "svd_fold_in"
        elif total > 0:
            stage = "tfidf_profile"
        else:
            stage = "cold_start"

        return {
            "stage": stage,
            "total_ratings": total,
            "svd_known_ratings": n_svd,
            "ratings_to_fold_in": max(0, self.svd_fold_in_threshold - n_svd),
            "svd_fold_in_active": stage == "svd_fold_in",
        }

    def search_by_text(
        self,
        query: str,
        top_n: int = 10,
        exclude: set | None = None,
    ) -> list[tuple[str, float]]:
        if self.tfidf_params is None:
            raise RuntimeError("tfidf_params not loaded — pass tfidf_params_path to load()")

        import json
        from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

        vectorizer = TfidfVectorizer(
            vocabulary=self.tfidf_params["vocabulary"],
            max_features=self.tfidf_params["max_features"],
            ngram_range=tuple(self.tfidf_params["ngram_range"]),
        )
        vectorizer.vocabulary_ = self.tfidf_params["vocabulary"]
        vectorizer._tfidf = TfidfTransformer()
        vectorizer._tfidf.idf_ = np.array(self.tfidf_params["idf"])

        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        exclude_set = set(exclude) if exclude else set()
        excluded_rows = {
            self.cid_to_tfidf_row[cid]
            for cid in exclude_set
            if cid in self.cid_to_tfidf_row
        }

        results = []
        for idx in np.argsort(sims)[::-1]:
            if idx not in excluded_rows:
                results.append((self.course_ids[idx], float(sims[idx])))
            if len(results) >= top_n:
                break
        return results

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _build_tfidf_profile(self, ratings: dict[str, float]) -> np.ndarray | None:
        qualifying = {
            cid: r
            for cid, r in ratings.items()
            if r >= self.rating_floor and cid in self.cid_to_tfidf_row
        }
        if not qualifying:
            return None

        rows = [self.cid_to_tfidf_row[cid] for cid in qualifying]
        weights = np.array([qualifying[cid] for cid in qualifying], dtype=float)
        vectors = self.tfidf_matrix[rows].toarray()
        return np.average(vectors, axis=0, weights=weights)

    def _fold_in_svd(
        self, ratings: dict[str, float]
    ) -> tuple[np.ndarray, float] | None:
        svd_ratings = {
            cid: r for cid, r in ratings.items() if cid in self.cid_to_svd_inner
        }
        if len(svd_ratings) < self.svd_fold_in_threshold:
            return None

        mu = self.svd.trainset.global_mean
        Q = self.svd.qi
        bi = self.svd.bi

        inner_ids = [self.cid_to_svd_inner[cid] for cid in svd_ratings]
        r = np.array([svd_ratings[cid] for cid in svd_ratings], dtype=float)
        bi_vals = bi[inner_ids]
        Q_rated = Q[inner_ids]

        b_u = float(np.mean(r - mu - bi_vals))
        d = r - mu - bi_vals - b_u

        n_factors = Q.shape[1]
        A = Q_rated.T @ Q_rated + self.reg * np.eye(n_factors)
        p_u = np.linalg.solve(A, Q_rated.T @ d)

        return p_u, b_u

    def _recommend_svd(
        self,
        p_u: np.ndarray,
        b_u: float,
        top_n: int,
        exclude: set,
    ) -> list[tuple[str, float]]:
        mu = self.svd.trainset.global_mean
        scores = np.clip(mu + b_u + self.svd.bi + self.svd.qi @ p_u, 1.0, 5.0)

        results = []
        for inner_id in np.argsort(scores)[::-1]:
            cid = self.svd_inner_to_cid[inner_id]
            if cid not in exclude:
                results.append((cid, float(scores[inner_id])))
            if len(results) >= top_n:
                break
        return results

    def _recommend_tfidf_profile(
        self,
        profile: np.ndarray,
        top_n: int,
        exclude: set,
    ) -> list[tuple[str, float]]:
        sims = cosine_similarity(profile.reshape(1, -1), self.tfidf_matrix).flatten()

        results = []
        for idx in np.argsort(sims)[::-1]:
            cid = self.course_ids[idx]
            if cid not in exclude:
                results.append((cid, float(sims[idx])))
            if len(results) >= top_n:
                break
        return results
