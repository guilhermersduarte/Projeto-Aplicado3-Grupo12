"""Metricas compartilhadas de avaliacao de recomendadores.

Todas as funcoes recebem a lista de predicoes Surprise (ou equivalente:
lista de tuplas/objetos com .uid, .iid, .r_ui, .est) e retornam um escalar.
"""

from collections import defaultdict

import numpy as np


def _unpack(pred):
    """Aceita tuplas (uid, iid, true_r, est, details) ou objetos Surprise."""
    if hasattr(pred, "uid"):
        return pred.uid, pred.iid, pred.r_ui, pred.est
    uid, iid, true_r, est = pred[0], pred[1], pred[2], pred[3]
    return uid, iid, true_r, est


def rmse(predictions):
    errs = [(_unpack(p)[2] - _unpack(p)[3]) ** 2 for p in predictions]
    return float(np.sqrt(np.mean(errs)))


def mae(predictions):
    errs = [abs(_unpack(p)[2] - _unpack(p)[3]) for p in predictions]
    return float(np.mean(errs))


def precision_at_k(predictions, k=10, threshold=4.0):
    """Proporcao de itens relevantes (r_ui >= threshold) no top-K por usuario."""
    user_est = defaultdict(list)
    for p in predictions:
        uid, _, true_r, est = _unpack(p)
        user_est[uid].append((est, true_r))

    precisions = []
    for ratings in user_est.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        n_rel = sum(1 for est, true_r in top_k if true_r >= threshold)
        precisions.append(n_rel / k)
    return float(np.mean(precisions))


def recall_at_k(predictions, k=10, threshold=4.0):
    """Fracao dos itens relevantes do usuario recuperados no top-K."""
    user_est = defaultdict(list)
    for p in predictions:
        uid, _, true_r, est = _unpack(p)
        user_est[uid].append((est, true_r))

    recalls = []
    for ratings in user_est.values():
        total_rel = sum(1 for _, true_r in ratings if true_r >= threshold)
        if total_rel == 0:
            continue
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        n_rel = sum(1 for est, true_r in top_k if true_r >= threshold)
        recalls.append(n_rel / total_rel)
    return float(np.mean(recalls)) if recalls else 0.0


def f1_at_k(predictions, k=10, threshold=4.0):
    p = precision_at_k(predictions, k, threshold)
    r = recall_at_k(predictions, k, threshold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(predictions, k=10, threshold=4.0):
    """Fracao de usuarios com ao menos 1 item relevante no top-K."""
    user_est = defaultdict(list)
    for p in predictions:
        uid, _, true_r, est = _unpack(p)
        user_est[uid].append((est, true_r))

    hits = 0
    for ratings in user_est.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        if any(true_r >= threshold for _, true_r in top_k):
            hits += 1
    return hits / len(user_est) if user_est else 0.0


def ndcg_at_k(predictions, k=10):
    """NDCG@K usando ganho = rating (1..5) e desconto log2.

    Para cada usuario, ordena por rating estimado e calcula DCG.
    IDCG e o DCG do ranking ideal (por r_ui). NDCG = DCG / IDCG.
    """
    user_est = defaultdict(list)
    for p in predictions:
        uid, _, true_r, est = _unpack(p)
        user_est[uid].append((est, true_r))

    def dcg(gains):
        return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ndcgs = []
    for ratings in user_est.values():
        if not ratings:
            continue
        by_est = sorted(ratings, key=lambda x: x[0], reverse=True)[:k]
        by_true = sorted(ratings, key=lambda x: x[1], reverse=True)[:k]
        gains_pred = [r for _, r in by_est]
        gains_ideal = [r for _, r in by_true]
        idcg = dcg(gains_ideal)
        if idcg == 0:
            continue
        ndcgs.append(dcg(gains_pred) / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _binarize(predictions, threshold=4.0):
    y_true, y_score = [], []
    for p in predictions:
        _, _, true_r, est = _unpack(p)
        y_true.append(1 if true_r >= threshold else 0)
        y_score.append(est)
    return np.array(y_true), np.array(y_score)


def auc_roc(predictions, threshold=4.0):
    """AUC-ROC com relevancia binaria (true_r >= threshold)."""
    from sklearn.metrics import roc_auc_score

    y_true, y_score = _binarize(predictions, threshold)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def roc_curve_data(predictions, threshold=4.0):
    from sklearn.metrics import roc_curve

    y_true, y_score = _binarize(predictions, threshold)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def pr_curve_data(predictions, threshold=4.0):
    from sklearn.metrics import precision_recall_curve

    y_true, y_score = _binarize(predictions, threshold)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return precision, recall


def evaluate_all(predictions, k_list=(5, 10), threshold=4.0):
    """Retorna dict com todas as metricas de uma vez."""
    out = {
        "rmse": rmse(predictions),
        "mae": mae(predictions),
        "auc_roc": auc_roc(predictions, threshold),
    }
    for k in k_list:
        out[f"precision@{k}"] = precision_at_k(predictions, k, threshold)
        out[f"recall@{k}"] = recall_at_k(predictions, k, threshold)
        out[f"f1@{k}"] = f1_at_k(predictions, k, threshold)
        out[f"hit_rate@{k}"] = hit_rate_at_k(predictions, k, threshold)
        out[f"ndcg@{k}"] = ndcg_at_k(predictions, k)
    return out
