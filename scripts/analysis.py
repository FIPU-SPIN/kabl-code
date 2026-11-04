#!/usr/bin/env python3
"""Inter-rater agreement analysis for KABL-ID.

Run after validate_csv.py and merge_codings.py:
    python scripts/analysis.py --n-boot 2000 --seed 42

Scales
  bloom : ordinal, 6 levels  -> Cohen's kappa (unweighted), quadratic-weighted kappa, Gwet AC1
  abc   : nominal, 6 types   -> Cohen's kappa, Gwet AC1
All pairwise statistics come with percentile-bootstrap 95% CIs (items resampled).
Krippendorff's alpha (nominal / ordinal / interval) for rater sets, also with bootstrap CIs.

Rater naming (from file names): humans = coder1, coder2 ; LLM = <model>_run1..3, <model>_shuffled
Model consensus = majority label over the model's base runs (run*); ties -> missing (counted).
Comparison types in the output:
  human_vs_human, llm_run_vs_human, llm_shuffled_vs_human, consensus_vs_human,
  llm_within_model (run vs run), shuffled_vs_base (order-sensitivity), consensus_vs_consensus
"""
import argparse
import itertools
import json
import platform
import warnings

import numpy as np
import pandas as pd

from common import ABC_TYPES, BLOOM_LEVELS, FIG_DIR, PROC_DIR, TABLES_DIR

SCALES = {
    "bloom": {"labels": BLOOM_LEVELS, "ordinal": True},
    "abc": {"labels": ABC_TYPES, "ordinal": False},
}


# ----------------------------------------------------------------- pairwise statistics
def conf_matrix(x, y, k):
    return np.bincount(x * k + y, minlength=k * k).reshape(k, k).astype(float)


def stats_from_cm(O, ordinal):
    """-> array [pct_agree, kappa, kappa_qw, ac1]"""
    k = O.shape[0]
    n = O.sum()
    if n == 0:
        return np.full(4, np.nan)
    r, c = O.sum(1), O.sum(0)
    po = np.trace(O) / n
    E = np.outer(r, c) / n
    pe = np.trace(E) / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    kqw = np.nan
    if ordinal:
        i, j = np.indices((k, k))
        W = (i - j) ** 2 / (k - 1) ** 2
        den = (W * E).sum()
        kqw = 1 - (W * O).sum() / den if den > 0 else np.nan
    pi = (r + c) / (2 * n)
    pe1 = (pi * (1 - pi)).sum() / (k - 1)
    ac1 = (po - pe1) / (1 - pe1) if pe1 < 1 else np.nan
    return np.array([po, kappa, kqw, ac1])


def pair_stats(x, y, k, ordinal, n_boot, rng):
    m = (x >= 0) & (y >= 0)
    xs, ys = x[m], y[m]
    n = len(xs)
    point = stats_from_cm(conf_matrix(xs, ys, k), ordinal)
    boots = np.empty((n_boot, 4))
    if n > 0:
        idx = rng.integers(0, n, size=(n_boot, n))
        for b in range(n_boot):
            boots[b] = stats_from_cm(conf_matrix(xs[idx[b]], ys[idx[b]], k), ordinal)
    else:
        boots[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
    return n, point, lo, hi


def landis_koch(v):
    if v is None or np.isnan(v):
        return ""
    if v < 0:
        return "poor"
    for thr, lab in [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if v <= thr:
            return lab
    return "almost perfect"


# ----------------------------------------------------------------- Krippendorff's alpha
def alpha_from_counts(C, mult, metric):
    """C: items x categories counts of ratings; mult: per-item bootstrap multiplicity."""
    m = C.sum(1)
    w = np.where(m >= 2, mult / np.maximum(m - 1, 1), 0.0)
    A = C * w[:, None]
    O = A.T @ C - np.diag(A.sum(0))
    nc = O.sum(1)
    n = nc.sum()
    if n <= 1:
        return np.nan
    k = len(nc)
    i, j = np.indices((k, k))
    if metric == "nominal":
        delta = (i != j).astype(float)
    elif metric == "interval":
        delta = ((i - j) ** 2).astype(float)
    elif metric == "ordinal":
        cs = np.cumsum(nc)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        S = cs[hi] - cs[lo] + nc[lo]
        delta = np.where(i == j, 0.0, (S - (nc[i] + nc[j]) / 2) ** 2)
    else:
        raise ValueError(metric)
    De = (np.outer(nc, nc) * delta).sum() / (n * (n - 1))
    if De == 0:
        return np.nan
    Do = (O * delta).sum() / n
    return 1 - Do / De


def counts_matrix(R, k):
    """R: raters x items, -1 = missing -> items x k counts."""
    C = np.zeros((R.shape[1], k))
    for u in range(R.shape[1]):
        v = R[:, u]
        v = v[v >= 0]
        C[u] = np.bincount(v, minlength=k)
    return C


def alpha_with_ci(R, k, metric, n_boot, rng):
    C = counts_matrix(R, k)
    N = C.shape[0]
    point = alpha_from_counts(C, np.ones(N), metric)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        boots[b] = alpha_from_counts(C, mult, metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


# ----------------------------------------------------------------- helpers
def encode_wide(wide, scale):
    labels = SCALES[scale]["labels"]
    out = {}
    for rid in wide.columns:
        col = wide[rid]
        if scale == "bloom":
            v = pd.to_numeric(col, errors="coerce")
            arr = np.where(v.isna(), -1, v.fillna(1).astype(int) - 1)
        else:
            mp = {l: i for i, l in enumerate(labels)}
            arr = col.map(mp).fillna(-1).astype(int).values
        out[rid] = arr.astype(int)
    return out


def majority(arrs, k):
    """Majority label across arrays (-1 = missing). Ties / no data -> -1."""
    M = np.vstack(arrs)
    out = np.full(M.shape[1], -1)
    n_tie = 0
    for u in range(M.shape[1]):
        v = M[:, u]
        v = v[v >= 0]
        if len(v) == 0:
            continue
        cnt = np.bincount(v, minlength=k)
        top = cnt.max()
        if (cnt == top).sum() == 1:
            out[u] = int(cnt.argmax())
        else:
            n_tie += 1
    return out, n_tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(PROC_DIR / "all_codings_long.csv", dtype={"grade": str})
    meta = long.drop_duplicates("rater_id").set_index("rater_id")[["rater_type", "model", "variant"]]
    wides = {s: pd.read_csv(PROC_DIR / f"{s}_wide.csv", dtype=str).drop(columns=["stage", "grade", "code"])
             for s in SCALES}
    wides = {s: w[meta.index] for s, w in wides.items()}
    keys = pd.read_csv(PROC_DIR / "bloom_wide.csv", dtype=str)[["stage", "grade", "code"]]

    humans = [r for r in meta.index if meta.loc[r, "rater_type"] == "human"]
    base = [r for r in meta.index if meta.loc[r, "rater_type"] == "llm" and meta.loc[r, "variant"].startswith("run")]
    shuf = [r for r in meta.index if meta.loc[r, "rater_type"] == "llm" and meta.loc[r, "variant"] == "shuffled"]
    models = sorted({meta.loc[r, "model"] for r in base + shuf})
    print(f"Humans: {humans}\nBase runs: {base}\nShuffled runs: {shuf}")

    # encode + consensus
    enc, cons_info = {}, []
    for s, cfg in SCALES.items():
        k = len(cfg["labels"])
        enc[s] = encode_wide(wides[s], s)
        for mdl in models:
            runs = [r for r in base if meta.loc[r, "model"] == mdl]
            if not runs:
                continue
            arr, n_tie = majority([enc[s][r] for r in runs], k)
            enc[s][f"{mdl}_consensus"] = arr
            cons_info.append({"model": mdl, "scale": s, "n_base_runs": len(runs),
                              "n_items_with_consensus": int((arr >= 0).sum()),
                              "n_ties_no_consensus": n_tie})
    pd.DataFrame(cons_info).to_csv(TABLES_DIR / "consensus_info.csv", index=False)
    cons = [f"{m}_consensus" for m in models if f"{m}_consensus" in enc["abc"]]
    model_of = {**{r: meta.loc[r, "model"] for r in meta.index},
                **{c: c.replace("_consensus", "") for c in cons}}

    # pair list
    pairs = []
    pairs += [("human_vs_human", a, b) for a, b in itertools.combinations(humans, 2)]
    pairs += [("llm_run_vs_human", a, h) for a in base for h in humans]
    pairs += [("llm_shuffled_vs_human", a, h) for a in shuf for h in humans]
    pairs += [("consensus_vs_human", c, h) for c in cons for h in humans]
    for m in models:
        rs = [r for r in base if meta.loc[r, "model"] == m]
        pairs += [("llm_within_model", a, b) for a, b in itertools.combinations(rs, 2)]
        pairs += [("shuffled_vs_base", s_, b) for s_ in shuf if meta.loc[s_, "model"] == m for b in rs]
    pairs += [("consensus_vs_consensus", a, b) for a, b in itertools.combinations(cons, 2)]

    rows, conf_rows = [], []
    for scale, cfg in SCALES.items():
        k, ordinal, labels = len(cfg["labels"]), cfg["ordinal"], cfg["labels"]
        for ctype, a, b in pairs:
            n, pt, lo, hi = pair_stats(enc[scale][a], enc[scale][b], k, ordinal, args.n_boot, rng)
            rows.append({
                "comparison_type": ctype, "scale": scale, "rater_a": a, "rater_b": b,
                "model_a": model_of.get(a, "human"), "model_b": model_of.get(b, "human"),
                "n_items": n, "pct_agreement": pt[0],
                "kappa": pt[1], "kappa_ci_lo": lo[1], "kappa_ci_hi": hi[1],
                "kappa_label": landis_koch(pt[1]),
                "kappa_qw": pt[2], "kappa_qw_ci_lo": lo[2], "kappa_qw_ci_hi": hi[2],
                "kappa_qw_label": landis_koch(pt[2]),
                "ac1": pt[3], "ac1_ci_lo": lo[3], "ac1_ci_hi": hi[3],
            })
            if ctype in ("human_vs_human", "llm_run_vs_human", "llm_shuffled_vs_human", "consensus_vs_human"):
                x, y = enc[scale][a], enc[scale][b]
                m = (x >= 0) & (y >= 0)
                O = conf_matrix(x[m], y[m], k)
                for i, j in zip(*np.nonzero(O)):
                    conf_rows.append({"comparison_type": ctype, "scale": scale, "rater_a": a, "rater_b": b,
                                      "label_a": labels[i], "label_b": labels[j], "count": int(O[i, j])})

    pw = pd.DataFrame(rows)
    num_cols = [c for c in pw.columns if pw[c].dtype == float]
    pw[num_cols] = pw[num_cols].round(4)
    pw.to_csv(TABLES_DIR / "pairwise_agreement.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(TABLES_DIR / "confusion_long.csv", index=False)

    # summary across pairs
    # consensus_vs_consensus is excluded: its pairs mix two different models
    summ = (pw[pw.comparison_type != "consensus_vs_consensus"]
              .groupby(["comparison_type", "scale", "model_a"])
              .agg(n_pairs=("kappa", "size"),
                   pct_agreement_mean=("pct_agreement", "mean"),
                   kappa_mean=("kappa", "mean"), kappa_min=("kappa", "min"), kappa_max=("kappa", "max"),
                   kappa_qw_mean=("kappa_qw", "mean"), kappa_qw_min=("kappa_qw", "min"),
                   kappa_qw_max=("kappa_qw", "max"), ac1_mean=("ac1", "mean"))
              .round(4).reset_index())
    summ.to_csv(TABLES_DIR / "summary_by_model.csv", index=False)

    # Krippendorff's alpha
    sets = {"humans_only": humans,
            "humans_plus_model_consensus": humans + cons,
            "humans_plus_all_base_runs": humans + base}
    arows = []
    for name, raters in sets.items():
        if len(raters) < 2:
            continue
        for scale, metrics in (("bloom", ["ordinal", "interval", "nominal"]), ("abc", ["nominal"])):
            k = len(SCALES[scale]["labels"])
            R = np.vstack([enc[scale][r] for r in raters])
            for metric in metrics:
                p, lo, hi = alpha_with_ci(R, k, metric, args.n_boot, rng)
                arows.append({"rater_set": name, "n_raters": len(raters), "scale": scale, "metric": metric,
                              "alpha": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
    pd.DataFrame(arows).to_csv(TABLES_DIR / "krippendorff_alpha.csv", index=False)

    # heatmap: humans + consensus (kappa_qw for Bloom, kappa for ABC)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ids = humans + cons
        if len(ids) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
            for ax, (scale, col, title) in zip(axes, [("bloom", "kappa_qw", "Bloom (weighted kappa)"),
                                                      ("abc", "kappa", "ABC (Cohen's kappa)")]):
                M = np.full((len(ids), len(ids)), np.nan)
                np.fill_diagonal(M, 1.0)
                sub = pw[(pw.scale == scale) & pw.rater_a.isin(ids) & pw.rater_b.isin(ids)]
                for _, r in sub.iterrows():
                    i, j = ids.index(r.rater_a), ids.index(r.rater_b)
                    M[i, j] = M[j, i] = r[col]
                im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
                ax.set_yticks(range(len(ids)), ids)
                for i in range(len(ids)):
                    for j in range(len(ids)):
                        if not np.isnan(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                    color="w" if M[i, j] < 0.6 else "k", fontsize=8)
                ax.set_title(title)
            fig.colorbar(im, ax=axes, shrink=0.8)
            fig.savefig(FIG_DIR / "agreement_heatmap.png", dpi=200, bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        print("Figure skipped:", e)

    (TABLES_DIR / "analysis_meta.json").write_text(json.dumps({
        "seed": args.seed, "n_boot": args.n_boot, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "raters": {"humans": humans, "base_runs": base, "shuffled_runs": shuf, "consensus": cons}},
        indent=2))
    print(f"Done. Tables in {TABLES_DIR}")


if __name__ == "__main__":
    main()
