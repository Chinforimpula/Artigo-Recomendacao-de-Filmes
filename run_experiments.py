#!/usr/bin/env python3
"""
Experimentos do artigo: filtragem colaborativa baseada em usuarios (KNN) no MovieLens 100K.

Uso:
    python3 run_experiments.py --data data/ml-100k --out results

Le u.data, u.item e as divisoes u1..u5 (base/test) e grava results/results.json.
Somente numpy e pandas. Sementes fixas para reproducibilidade.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SIMS = ["cosine", "pearson", "msd"]
MODES = ["basic", "centered"]
K_GRID = [5, 10, 20, 30, 40, 50, 60, 80, 100]
MIN_SUPPORT = 5
MIN_SUPPORT_GRID = [2, 5, 10, 20, 40]
TOPN = 10
REL_THRESHOLD = 4
MIN_NB_LIST = [1, 5, 20]
GROUP_EDGES = [0, 30, 60, 120, 10 ** 9]
GROUP_LABELS = ["<30", "30-59", "60-119", ">=120"]
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]


# ------------------------------------------------------------------ dados
def load_ratings(path):
    return pd.read_csv(path, sep="\t", names=["u", "i", "r", "t"], engine="python")


def to_matrix(df, n_users, n_items):
    R = np.zeros((n_users, n_items), dtype=float)
    R[df["u"].values - 1, df["i"].values - 1] = df["r"].values
    return R


def group_of(n):
    for g in range(len(GROUP_LABELS)):
        if GROUP_EDGES[g] <= n < GROUP_EDGES[g + 1]:
            return g
    return len(GROUP_LABELS) - 1


def dataset_stats(df, n_users, n_items):
    cnt_u = df.groupby("u").size()
    cnt_i = df.groupby("i").size()
    dist = df["r"].value_counts().sort_index()
    sorted_i = np.sort(cnt_i.values)[::-1]
    top20 = int(np.ceil(0.2 * len(sorted_i)))
    return {
        "n_ratings": int(len(df)),
        "n_users": int(df["u"].nunique()),
        "n_items_rated": int(df["i"].nunique()),
        "n_items_catalog": int(n_items),
        "density": float(len(df) / (n_users * n_items)),
        "mean_rating": float(df["r"].mean()),
        "std_rating": float(df["r"].std()),
        "rating_dist": {int(k): int(v) for k, v in dist.items()},
        "ratings_per_user": {"min": int(cnt_u.min()), "median": float(cnt_u.median()),
                             "mean": float(cnt_u.mean()), "max": int(cnt_u.max())},
        "ratings_per_item": {"min": int(cnt_i.min()), "median": float(cnt_i.median()),
                             "mean": float(cnt_i.mean()), "max": int(cnt_i.max())},
        "share_items_lt_10": float((cnt_i < 10).mean()),
        "share_ratings_top20pct_items": float(sorted_i[:top20].sum() / sorted_i.sum()),
        "ratings_per_user_values": [int(x) for x in cnt_u.values],
        "ratings_per_item_values": [int(x) for x in cnt_i.values],
    }


def load_items(path, n_items):
    titles, genres = {}, {}
    if not Path(path).exists():
        return titles, genres
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            p = line.rstrip("\n").split("|")
            if len(p) < 24:
                continue
            iid = int(p[0])
            titles[iid] = p[1]
            genres[iid] = [GENRES[j] for j, f in enumerate(p[5:24]) if f == "1"]
    return titles, genres


# ------------------------------------------------------------- similaridade
def similarity(R, kind, min_support):
    """Similaridade usuario-usuario sobre os itens avaliados por ambos (co-avaliados)."""
    M = (R > 0).astype(float)
    n = M @ M.T
    Sxy = R @ R.T
    Sxx = (R * R) @ M.T          # [u,v] = soma sobre co-avaliados de r_ui^2
    Syy = Sxx.T
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "cosine":
            den = np.sqrt(Sxx * Syy)
            S = np.where(den > 0, Sxy / den, 0.0)
        elif kind == "pearson":
            Sx = R @ M.T
            Sy = Sx.T
            nn = np.maximum(n, 1.0)
            cov = Sxy - Sx * Sy / nn
            vx = np.maximum(Sxx - Sx ** 2 / nn, 0.0)
            vy = np.maximum(Syy - Sy ** 2 / nn, 0.0)
            den = np.sqrt(vx * vy)
            S = np.where(den > 1e-12, cov / den, 0.0)
            S = np.clip(S, -1.0, 1.0)
        elif kind == "msd":
            msd = (Sxx + Syy - 2.0 * Sxy) / np.maximum(n, 1.0)
            S = 1.0 / (1.0 + np.maximum(msd, 0.0))
        else:
            raise ValueError(kind)
    S = np.nan_to_num(S)
    S[n < max(min_support, 1)] = 0.0
    np.fill_diagonal(S, 0.0)
    return S


# --------------------------------------------------------------- previsao
def user_means(R, global_mean):
    M = R > 0
    cnt = M.sum(1)
    return np.where(cnt > 0, R.sum(1) / np.maximum(cnt, 1), global_mean)


def predict_pairs(S, R, mu, pu, pi, ks):
    """Previsoes para pares (usuario,item) em varios k de uma vez.
    Retorna dict modo -> array (len(ks), n_pares) e cobertura (len(ks), n_pares)."""
    M = R > 0
    raters = [np.nonzero(M[:, i])[0] for i in range(R.shape[1])]
    kmax = max(ks)
    out = {m: np.zeros((len(ks), len(pu))) for m in MODES}
    cov = np.zeros((len(ks), len(pu)), dtype=bool)
    for idx in range(len(pu)):
        u, i = pu[idx], pi[idx]
        v = raters[i]
        s = S[u, v]
        pos = s > 0
        v, s = v[pos], s[pos]
        if len(v) == 0:
            for m in MODES:
                out[m][:, idx] = mu[u]
            continue
        order = np.argsort(-s, kind="stable")[:kmax]
        v, s = v[order], s[order]
        r = R[v, i]
        cs = np.cumsum(s)
        csr = np.cumsum(s * r)
        csc = np.cumsum(s * (r - mu[v]))
        for a, k in enumerate(ks):
            q = min(k, len(v)) - 1
            out["basic"][a, idx] = csr[q] / cs[q]
            out["centered"][a, idx] = mu[u] + csc[q] / cs[q]
            cov[a, idx] = True
    for m in MODES:
        out[m] = np.clip(out[m], 1.0, 5.0)
    return out, cov


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def group_rmse(users_ntrain, y, p):
    g = np.array([group_of(n) for n in users_ntrain])
    res = {}
    for gi, lab in enumerate(GROUP_LABELS):
        m = g == gi
        res[lab] = rmse(y[m], p[m]) if m.sum() > 0 else None
    return res


# ----------------------------------------------------------------- top-N
def score_all_items(u, S, R, M, mu, k, mode):
    """Previsao do usuario u para todos os itens; devolve (pred, n_vizinhos)."""
    s = S[u]
    W = np.where(M & (s[:, None] > 0), s[:, None], 0.0)
    if W.shape[0] > k:
        thr = np.partition(W, -k, axis=0)[-k]
        W = np.where(W >= thr[None, :], W, 0.0)
    den = W.sum(0)
    cnt = (W > 0).sum(0)
    if mode == "basic":
        num = (W * R).sum(0)
        base = 0.0
    else:
        num = (W * (R - mu[:, None])).sum(0)
        base = mu[u]
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.where(den > 0, base + num / den, np.nan)
    return pred, cnt


def topn_user(score, tie, cand, n=TOPN):
    idx = np.nonzero(cand & ~np.isnan(score))[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    order = np.lexsort((-tie[idx], -score[idx]))
    return idx[order][:n]


def evaluate_topn_fold(S, R, test_df, k, mode, mu):
    M = R > 0
    n_users, n_items = R.shape
    item_cnt = M.sum(0).astype(float)
    item_mean = np.where(item_cnt > 0, R.sum(0) / np.maximum(item_cnt, 1), 0.0)
    pop_score = item_cnt
    best_score = np.where(item_cnt >= 20, item_mean, -np.inf)
    rel = {}
    for u, i, r in zip(test_df["u"].values - 1, test_df["i"].values - 1, test_df["r"].values):
        if r >= REL_THRESHOLD:
            rel.setdefault(u, set()).add(i)
    methods = ["knn_m%d" % m for m in MIN_NB_LIST] + ["popular", "topmean"]
    rows = {m: [] for m in methods}
    for u in sorted(rel.keys()):
        cand = ~M[u]
        pred, cnt = score_all_items(u, S, R, M, mu, k, mode)
        lists = {}
        for m in MIN_NB_LIST:
            sc = np.where(cnt >= m, pred, np.nan)
            lists["knn_m%d" % m] = topn_user(sc, cnt.astype(float), cand)
        lists["popular"] = topn_user(pop_score.astype(float), pop_score, cand)
        bs = np.where(np.isfinite(best_score), best_score, np.nan)
        lists["topmean"] = topn_user(bs, item_cnt, cand)
        ntr = int(M[u].sum())
        for m in methods:
            hits = len(set(lists[m].tolist()) & rel[u])
            rows[m].append((u, ntr, hits / TOPN, hits / len(rel[u])))
    return rows


def aggregate_rows(rows):
    arr = np.array(rows)
    out = {"n_users": int(len(arr)), "precision": float(arr[:, 2].mean()),
           "recall": float(arr[:, 3].mean()), "groups": {}}
    g = np.array([group_of(n) for n in arr[:, 1]])
    for gi, lab in enumerate(GROUP_LABELS):
        m = g == gi
        out["groups"][lab] = {"n_users": int(m.sum()),
                              "precision": float(arr[m, 2].mean()) if m.sum() else None,
                              "recall": float(arr[m, 3].mean()) if m.sum() else None}
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta ml-100k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="teste rapido do pipeline")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [5, 20, 50] if a.quick else K_GRID
    t0 = time.time()

    full = load_ratings(data / "u.data")
    n_users, n_items = int(full["u"].max()), int(full["i"].max())
    titles, genres = load_items(data / "u.item", n_items)
    res = {"config": {"seed": SEED, "sims": SIMS, "modes": MODES, "k_grid": ks,
                      "min_support": MIN_SUPPORT, "topn": TOPN, "rel_threshold": REL_THRESHOLD,
                      "min_nb_list": MIN_NB_LIST, "group_labels": GROUP_LABELS,
                      "n_users": n_users, "n_items": n_items, "folds": a.folds},
           "dataset": dataset_stats(full, n_users, n_items), "folds": []}

    # ---------------- passo 1: previsao de notas
    store = {}
    for f in range(1, a.folds + 1):
        base = load_ratings(data / ("u%d.base" % f))
        test = load_ratings(data / ("u%d.test" % f))
        R = to_matrix(base, n_users, n_items)
        gm = float(base["r"].mean())
        mu = user_means(R, gm)
        tu, ti, ty = test["u"].values - 1, test["i"].values - 1, test["r"].values.astype(float)
        ntr = (R > 0).sum(1)[tu]

        rng = np.random.RandomState(SEED + f)
        perm = rng.permutation(len(base))
        nval = int(0.1 * len(base))
        val, inner = base.iloc[perm[:nval]], base.iloc[perm[nval:]]
        Ri = to_matrix(inner, n_users, n_items)
        mui = user_means(Ri, float(inner["r"].mean()))
        vu, vi, vy = val["u"].values - 1, val["i"].values - 1, val["r"].values.astype(float)

        fold = {"fold": f, "n_train": int(len(base)), "n_test": int(len(test)), "knn": {},
                "baselines": {}}
        # baselines
        item_cnt = (R > 0).sum(0)
        item_mean = np.where(item_cnt > 0, R.sum(0) / np.maximum(item_cnt, 1), gm)
        for name, p in [("global_mean", np.full(len(ty), gm)), ("user_mean", mu[tu]),
                        ("item_mean", item_mean[ti])]:
            fold["baselines"][name] = {"rmse": rmse(ty, p), "mae": mae(ty, p),
                                       "group_rmse": group_rmse(ntr, ty, p)}
        for sim in SIMS:
            Si = similarity(Ri, sim, MIN_SUPPORT)
            pv, _ = predict_pairs(Si, Ri, mui, vu, vi, ks)
            Sf = similarity(R, sim, MIN_SUPPORT)
            pt, cov = predict_pairs(Sf, R, mu, tu, ti, ks)
            for mode in MODES:
                val_r = [rmse(vy, pv[mode][a_]) for a_ in range(len(ks))]
                astar = int(np.argmin(val_r))
                p = pt[mode][astar]
                fold["knn"]["%s|%s" % (sim, mode)] = {
                    "val_rmse_by_k": val_r, "k_star": ks[astar], "val_rmse": val_r[astar],
                    "test_rmse_by_k": [rmse(ty, pt[mode][a_]) for a_ in range(len(ks))],
                    "test_mae_by_k": [mae(ty, pt[mode][a_]) for a_ in range(len(ks))],
                    "coverage_by_k": [float(cov[a_].mean()) for a_ in range(len(ks))],
                    "test_rmse": rmse(ty, p), "test_mae": mae(ty, p),
                    "coverage": float(cov[astar].mean()), "group_rmse": group_rmse(ntr, ty, p)}
        res["folds"].append(fold)
        store[f] = (base, test)
        print("fold %d ok (%.0fs)" % (f, time.time() - t0), flush=True)

    # configuracao vencedora: menor RMSE medio de VALIDACAO (nunca o teste)
    cfgs = ["%s|%s" % (s, m) for s in SIMS for m in MODES]
    val_mean = {c: float(np.mean([fo["knn"][c]["val_rmse"] for fo in res["folds"]])) for c in cfgs}
    best = min(val_mean, key=val_mean.get)
    res["best_config"] = best
    res["val_rmse_mean"] = val_mean
    bsim, bmode = best.split("|")
    print("config vencedora (validacao):", best, flush=True)

    # ---------------- passo 2: top-N, ablacao de suporte minimo, exemplo qualitativo
    res["topn"] = []
    res["support_ablation"] = []
    for f in range(1, a.folds + 1):
        base, test = store[f]
        R = to_matrix(base, n_users, n_items)
        gm = float(base["r"].mean())
        mu = user_means(R, gm)
        kstar = res["folds"][f - 1]["knn"][best]["k_star"]
        S = similarity(R, bsim, MIN_SUPPORT)
        rows = evaluate_topn_fold(S, R, test, kstar, bmode, mu)
        res["topn"].append({m: aggregate_rows(r) for m, r in rows.items()})
        tu, ti, ty = test["u"].values - 1, test["i"].values - 1, test["r"].values.astype(float)
        abl = {}
        for ms in MIN_SUPPORT_GRID:
            Sm = similarity(R, bsim, ms)
            pt, cov = predict_pairs(Sm, R, mu, tu, ti, [kstar])
            abl[str(ms)] = {"rmse": rmse(ty, pt[bmode][0]), "mae": mae(ty, pt[bmode][0]),
                            "coverage": float(cov[0].mean())}
        res["support_ablation"].append(abl)
        print("top-N/ablacao fold %d ok (%.0fs)" % (f, time.time() - t0), flush=True)

    # exemplo qualitativo: modelo treinado com todos os dados, usuario com historico mediano
    Rf = to_matrix(full, n_users, n_items)
    muf = user_means(Rf, float(full["r"].mean()))
    Sf = similarity(Rf, bsim, MIN_SUPPORT)
    kmode = int(np.median([fo["knn"][best]["k_star"] for fo in res["folds"]]))
    cnt_u = (Rf > 0).sum(1)
    u = int(np.argmin(np.abs(cnt_u - np.median(cnt_u))))
    Mf = Rf > 0
    pred, cnt = score_all_items(u, Sf, Rf, Mf, muf, kmode, bmode)
    sc = np.where(cnt >= 5, pred, np.nan)
    rec = topn_user(sc, cnt.astype(float), ~Mf[u], n=5)
    liked = np.argsort(-(Rf[u] + 1e-3 * np.arange(n_items)[::-1] / n_items))[:5]
    res["example"] = {
        "user_id": u + 1, "n_ratings": int(cnt_u[u]), "k": kmode, "config": best,
        "liked": [{"title": titles.get(int(i) + 1, str(i + 1)), "rating": float(Rf[u, i]),
                   "genres": genres.get(int(i) + 1, [])} for i in liked],
        "recs": [{"title": titles.get(int(i) + 1, str(i + 1)), "pred": float(min(pred[i], 5.0)),
                  "neighbors": int(cnt[i]), "genres": genres.get(int(i) + 1, [])} for i in rec]}

    res["runtime_seconds"] = time.time() - t0
    with open(out / "results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("pronto: %s (%.0fs)" % (out / "results.json", time.time() - t0))


if __name__ == "__main__":
    main()
