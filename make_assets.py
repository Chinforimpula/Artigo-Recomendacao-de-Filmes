#!/usr/bin/env python3
"""Gera figuras (PDF vetorial), tabelas e macros LaTeX a partir de results/results.json.
Uso: python3 make_assets.py --results results/results.json --out artigo_assets
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM_NAME = {"cosine": "Cosseno", "pearson": "Pearson", "msd": "MSD"}
MODE_NAME = {"basic": "B\\'asico", "centered": "Centralizado"}
COLORS = {"cosine": "#1F4E79", "pearson": "#C0504D", "msd": "#4F9D5D"}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})


def num(x, d=3):
    return ("%.*f" % (d, x)).replace(".", ",")


def pm(vals, d=3):
    return "%s $\\pm$ %s" % (num(np.mean(vals), d), num(np.std(vals, ddof=1), d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(a.results))
    out = Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    folds, ds, cfg = res["folds"], res["dataset"], res["config"]
    ks = cfg["k_grid"]
    best = res["best_config"]
    bsim, bmode = best.split("|")
    nf = len(folds)

    # ---------- macros numericos
    rpu, rpi = ds["ratings_per_user"], ds["ratings_per_item"]
    best_rm = [f["knn"][best]["test_rmse"] for f in folds]
    best_mae = [f["knn"][best]["test_mae"] for f in folds]
    base_names = ["global_mean", "user_mean", "item_mean"]
    base_rm = {b: [f["baselines"][b]["rmse"] for f in folds] for b in base_names}
    wins = {b: int(sum(best_rm[i] < base_rm[b][i] for i in range(nf))) for b in base_names}
    kstars = [f["knn"][best]["k_star"] for f in folds]
    M = {
        "nRatings": "{:,}".format(ds["n_ratings"]).replace(",", "."),
        "nUsers": "{:,}".format(ds["n_users"]).replace(",", "."),
        "nItems": "{:,}".format(ds["n_items_catalog"]).replace(",", "."),
        "density": num(100 * ds["density"], 2),
        "emptyShare": num(100 * (1 - ds["density"]), 1),
        "meanRating": num(ds["mean_rating"], 2),
        "stdRating": num(ds["std_rating"], 2),
        "rpuMin": str(rpu["min"]), "rpuMed": num(rpu["median"], 0), "rpuMean": num(rpu["mean"], 0),
        "rpuMax": str(rpu["max"]),
        "rpiMed": num(rpi["median"], 0), "rpiMean": num(rpi["mean"], 0), "rpiMax": str(rpi["max"]),
        "itemsLtTen": num(100 * ds["share_items_lt_10"], 0),
        "topTwentyShare": num(100 * ds["share_ratings_top20pct_items"], 0),
        "bestSim": SIM_NAME[bsim], "bestMode": "centralizado" if bmode == "centered" else "b\\'asico",
        "bestRMSE": num(np.mean(best_rm)), "bestRMSEstd": num(np.std(best_rm, ddof=1)),
        "bestMAE": num(np.mean(best_mae)),
        "bestK": str(int(np.median(kstars))),
        "winsGlobal": str(wins["global_mean"]), "winsUser": str(wins["user_mean"]),
        "winsItem": str(wins["item_mean"]), "nFolds": str(nf),
        "baseGlobal": num(np.mean(base_rm["global_mean"])),
        "baseUser": num(np.mean(base_rm["user_mean"])),
        "baseItem": num(np.mean(base_rm["item_mean"])),
        "gainItemPct": num(100 * (np.mean(base_rm["item_mean"]) - np.mean(best_rm))
                           / np.mean(base_rm["item_mean"]), 1),
        "gainUserPct": num(100 * (np.mean(base_rm["user_mean"]) - np.mean(best_rm))
                           / np.mean(base_rm["user_mean"]), 1),
        "gainGlobalPct": num(100 * (np.mean(base_rm["global_mean"]) - np.mean(best_rm))
                             / np.mean(base_rm["global_mean"]), 1),
        "runtime": num(res["runtime_seconds"] / 60, 0),
    }
    with open(out / "numbers.tex", "w") as fh:
        for k, v in M.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

    # ---------- Fig. 1: dataset
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.7))
    d = ds["rating_dist"]
    xs = sorted(int(k) for k in d)
    tot = sum(d.values())
    ax[0].bar(xs, [100 * d[str(x)] / tot for x in xs], color="#1F4E79", width=0.7)
    ax[0].set_xticks(xs); ax[0].set_xlabel("Nota"); ax[0].set_ylabel("% das avalia\u00e7\u00f5es")
    ax[0].set_title("(a) Distribui\u00e7\u00e3o das notas", fontsize=7)
    cnts = np.sort(np.array(ds["ratings_per_item_values"]))[::-1]
    ax[1].plot(np.arange(1, len(cnts) + 1), cnts, color="#C0504D", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Posi\u00e7\u00e3o do filme"); ax[1].set_ylabel("N\u00ba de avalia\u00e7\u00f5es")
    ax[1].set_title("(b) Cauda longa", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "figs/fig_dataset.pdf"); plt.close(fig)

    # ---------- Fig. 2: RMSE x k
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for sim in cfg["sims"]:
        for mode in cfg["modes"]:
            curve = np.mean([f["knn"]["%s|%s" % (sim, mode)]["test_rmse_by_k"] for f in folds], axis=0)
            ax.plot(ks, curve, color=COLORS[sim], ls="-" if mode == "centered" else "--",
                    marker="o", ms=2.2, lw=1,
                    label="%s (%s)" % (SIM_NAME[sim], "centr." if mode == "centered" else "b\u00e1sico"))
    ax.axhline(np.mean(base_rm["item_mean"]), color="gray", ls=":", lw=1, label="M\u00e9dia do filme")
    ax.set_xlabel("N\u00famero de vizinhos $k$"); ax.set_ylabel("RMSE (teste)")
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "figs/fig_rmse_k.pdf"); plt.close(fig)

    # ---------- Tabela I: similaridade x modo
    rows = []
    for sim in cfg["sims"]:
        for mode in cfg["modes"]:
            c = "%s|%s" % (sim, mode)
            rm = [f["knn"][c]["test_rmse"] for f in folds]
            ma = [f["knn"][c]["test_mae"] for f in folds]
            kk = int(np.median([f["knn"][c]["k_star"] for f in folds]))
            cv = 100 * np.mean([f["knn"][c]["coverage"] for f in folds])
            bold = "\\textbf{%s}" if c == best else "%s"
            rows.append("%s & %s & %s & %s & %d & %s \\\\" % (
                SIM_NAME[sim], MODE_NAME[mode], bold % pm(rm), pm(ma), kk, num(cv, 1)))
    t1 = ("\\begin{tabular}{llcccc}\n\\hline\nSimilaridade & Modo & RMSE & MAE & $k$ & Cob. (\\%) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    (out / "table_sims.tex").write_text(t1)

    # ---------- Tabela II: baselines
    names = {"global_mean": "M\\'edia global", "user_mean": "M\\'edia do usu\\'ario",
             "item_mean": "M\\'edia do filme"}
    rows = []
    for b in base_names:
        rows.append("%s & %s & %s \\\\" % (names[b], pm(base_rm[b]),
                                          pm([f["baselines"][b]["mae"] for f in folds])))
    rows.append("\\textbf{KNN (%s, %s)} & \\textbf{%s} & \\textbf{%s} \\\\" % (
        SIM_NAME[bsim], "centr." if bmode == "centered" else "b\\'asico", pm(best_rm), pm(best_mae)))
    t2 = ("\\begin{tabular}{lcc}\n\\hline\nModelo & RMSE & MAE \\\\\n\\hline\n" + "\n".join(rows)
          + "\n\\hline\n\\end{tabular}\n")
    (out / "table_baselines.tex").write_text(t2)

    # ---------- Tabela III: top-N geral e Fig. 3 por grupo
    meth = [("knn_m1", "KNN ($\\geq$1 vizinho)"), ("knn_m5", "KNN ($\\geq$5 vizinhos)"),
            ("knn_m20", "KNN ($\\geq$20 vizinhos)"),
            ("popular", "Mais populares"), ("topmean", "Melhor m\\'edia ($\\geq$20 aval.)")]
    rows = []
    for key, lab in meth:
        p = [f[key]["precision"] for f in res["topn"]]
        r = [f[key]["recall"] for f in res["topn"]]
        rows.append("%s & %s & %s \\\\" % (lab, pm(p), pm(r)))
    t3 = ("\\begin{tabular}{lcc}\n\\hline\nM\\'etodo & Precision@10 & Recall@10 \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    (out / "table_topn.tex").write_text(t3)

    labs = cfg["group_labels"]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    w = 0.26
    for j, (key, lab) in enumerate([("knn_m20", "KNN ($\\geq$20 viz.)"), ("popular", "Mais populares"),
                                    ("topmean", "Melhor m\u00e9dia")]):
        mean, err = [], []
        for g in labs:
            v = [f[key]["groups"][g]["precision"] for f in res["topn"]
                 if f[key]["groups"][g]["precision"] is not None]
            mean.append(np.mean(v)); err.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        ax.bar(np.arange(len(labs)) + (j - 1) * w, mean, w, yerr=err, capsize=1.5,
               color=["#C0504D", "#1F4E79", "#4F9D5D"][j], label=lab, error_kw={"lw": 0.6})
    ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs)
    ax.set_xlabel("N\u00ba de avalia\u00e7\u00f5es do usu\u00e1rio no treino"); ax.set_ylabel("P@10")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "figs/fig_groups.pdf"); plt.close(fig)

    # ---------- Tabela IV: RMSE por grupo
    rows = []
    for g in labs:
        def gm(getter):
            v = [getter(f)[g] for f in folds if getter(f)[g] is not None]
            return np.mean(v)
        n_users = int(np.mean([f_["knn_m5"]["groups"][g]["n_users"] for f_ in res["topn"]]))
        rows.append("%s & %s & %s & %s \\\\" % (
            g, num(gm(lambda f: f["knn"][best]["group_rmse"])),
            num(gm(lambda f: f["baselines"]["item_mean"]["group_rmse"])),
            num(gm(lambda f: f["baselines"]["user_mean"]["group_rmse"]))))
    t4 = ("\\begin{tabular}{lccc}\n\\hline\nAvalia\\c{c}\\~oes & KNN & M\\'edia filme & M\\'edia usu\\'ario \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    (out / "table_groups.tex").write_text(t4)

    # ---------- Tabela V: suporte minimo
    rows = []
    for ms in [str(x) for x in sorted(int(k) for k in res["support_ablation"][0])]:
        rm = [f[ms]["rmse"] for f in res["support_ablation"]]
        cv = 100 * np.mean([f[ms]["coverage"] for f in res["support_ablation"]])
        rows.append("%s & %s & %s \\\\" % (ms, pm(rm), num(cv, 1)))
    t5 = ("\\begin{tabular}{lcc}\n\\hline\nM\\'inimo de filmes em comum & RMSE & Cob. (\\%) \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    (out / "table_support.tex").write_text(t5)

    # ---------- Tabela VI: exemplo
    ex = res["example"]
    def esc(s):
        return s.replace("&", "\\&").replace("_", "\\_").replace("#", "\\#").replace("%", "\\%")
    rows = ["%s & %s & %d \\\\" % (esc(r["title"]), num(r["pred"], 2), r["neighbors"]) for r in ex["recs"]]
    t6 = ("\\begin{tabular}{p{1.95in}cc}\n\\hline\nFilme recomendado & Nota prevista & Vizinhos \\\\\n\\hline\n"
          + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
    (out / "table_example.tex").write_text(t6)
    liked = "; ".join("%s (%d)" % (esc(x["title"]), int(x["rating"])) for x in ex["liked"])
    (out / "example_liked.tex").write_text(liked)
    (out / "example_meta.tex").write_text(
        "\\newcommand{\\exUser}{%d}\n\\newcommand{\\exN}{%d}\n\\newcommand{\\exK}{%d}\n"
        % (ex["user_id"], ex["n_ratings"], ex["k"]))
    print("assets em", out)


if __name__ == "__main__":
    main()
