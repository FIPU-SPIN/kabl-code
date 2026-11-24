#!/usr/bin/env python3
"""KABL-ID additional statistics, tables, and figures (v6: fixed figures)."""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BLOOM_LABELS = ["1\nRemember", "2\nUnderstand", "3\nApply",
                "4\nAnalyze", "5\nEvaluate", "6\nCreate"]
ABC_LABELS = ["Acquisition", "Investigation", "Practice",
              "Discussion", "Collaboration", "Production"]
MODELS = ["chatgpt", "claude", "deepseek", "gemini", "perplexity"]

C1, C2, CLLM = "#5B4B8A", "#E07A5F", "#3D9B8F"
CMODELS = {"chatgpt": "#5B4B8A", "claude": "#E07A5F", "deepseek": "#3D9B8F",
           "gemini": "#8B5A2B", "perplexity": "#7B6D8D"}
CEDGE = "#222222"
CSHIFT = "#F4A6A6"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": CEDGE, "axes.linewidth": 0.9,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "figure.autolayout": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ================================================================== load
bloom = pd.read_csv(PROC / "bloom_wide.csv", dtype={"grade": str})
abc = pd.read_csv(PROC / "abc_wide.csv", dtype={"grade": str})
long = pd.read_csv(PROC / "all_codings_long.csv", dtype={"grade": str})

base_cols = {m: [c for c in bloom.columns if c.startswith(m + "_run")] for m in MODELS}


def majority_row(row, is_bloom=True):
    vals = row.dropna().tolist()
    if is_bloom:
        vals = [int(v) for v in vals if pd.notna(v)]
    if not vals:
        return np.nan
    s = pd.Series(vals)
    mode = s.mode()
    return mode.iloc[0] if len(mode) == 1 else np.nan


for df, is_b in [(bloom, True), (abc, False)]:
    for m in MODELS:
        if base_cols[m]:
            df[f"{m}_consensus"] = df[base_cols[m]].apply(
                lambda r: majority_row(r, is_bloom=is_b), axis=1)

all_base = [c for m in MODELS for c in base_cols[m]]
bloom["llm_consensus"] = bloom[all_base].apply(lambda r: majority_row(r, True), axis=1)
abc["llm_consensus"] = abc[all_base].apply(lambda r: majority_row(r, False), axis=1)
bloom["domain"] = bloom["code"].str[0]


# ================================================================== helpers
def as_num(series):
    return pd.to_numeric(series, errors="coerce")


def counts(series, labels):
    s = series.dropna()
    if labels[0].startswith("1"):
        s = s.astype(int)
        c = s.value_counts().reindex(range(1, 7), fill_value=0)
        c.index = [str(i) for i in range(1, 7)]
    else:
        c = s.value_counts().reindex(labels, fill_value=0)
    return c


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) == 0:
        return np.nan
    ct = pd.crosstab(df["a"], df["b"])
    n = ct.values.sum()
    po = np.trace(ct.values) / n
    pe = (ct.sum(1).values @ ct.sum(0).values) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def weighted_kappa(a, b, k=6):
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(df) == 0:
        return np.nan
    ct = pd.crosstab(df["a"].astype(int), df["b"].astype(int), dropna=False
                     ).reindex(index=range(1, k + 1), columns=range(1, k + 1),
                               fill_value=0).values
    n = ct.sum()
    if n == 0:
        return np.nan
    r, c = ct.sum(1), ct.sum(0)
    E = np.outer(r, c) / n
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2
    den = (W * E).sum()
    return np.nan if den == 0 else 1 - (W * ct).sum() / den


# ================================================================== TABLES
c1_b = counts(bloom["coder1"], BLOOM_LABELS)
c2_b = counts(bloom["coder2"], BLOOM_LABELS)
cl_b = counts(bloom["llm_consensus"], BLOOM_LABELS)

rows = []
for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                  ("llm_consensus", "llm_consensus")]:
    s = as_num(bloom[col]).dropna()
    if len(s) == 0:
        continue
    n = len(s)
    cnts = [int((s == i).sum()) for i in range(1, 7)]
    shift_pct = 100.0 * (s >= 4).sum() / n
    rows.append(f"{name} & {n} & " + " & ".join(str(c) for c in cnts)
                + f" & {shift_pct:.1f} \\\\")
(TAB / "tab_bloom_distribution.txt").write_text("\n".join(rows), encoding="utf-8")

c1_a = counts(abc["coder1"], ABC_LABELS)
c2_a = counts(abc["coder2"], ABC_LABELS)
cl_a = counts(abc["llm_consensus"], ABC_LABELS)

rows = []
for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                  ("llm_consensus", "llm_consensus")]:
    s = abc[col].dropna()
    if len(s) == 0:
        continue
    n = len(s)
    cnts = [int((s == lab).sum()) for lab in ABC_LABELS]
    rows.append(f"{name} & {n} & " + " & ".join(str(c) for c in cnts) + " \\\\")
(TAB / "tab_abc_distribution.txt").write_text("\n".join(rows), encoding="utf-8")

crosstab = pd.crosstab(as_num(bloom["llm_consensus"]).dropna().astype(int),
                       abc["llm_consensus"].dropna())
crosstab = crosstab.reindex(index=[1, 2, 3, 4, 5, 6], columns=ABC_LABELS, fill_value=0)
crosstab.to_csv(TAB / "tab_bloom_x_abc.csv")

shift_rows = []
for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                  ("LLM consensus", "llm_consensus")] + \
                 [(m, f"{m}_consensus") for m in MODELS]:
    s = as_num(bloom[col]).dropna()
    if len(s) == 0:
        continue
    n = len(s)
    k = int((s >= 4).sum())
    lo, hi = wilson_ci(k, n)
    shift_rows.append(f"{name} & {k} & {n} & {100.0 * k / n:.1f} "
                      f"[{100.0 * lo:.1f}, {100.0 * hi:.1f}] \\\\")
(TAB / "tab_shift_sensitive.txt").write_text("\n".join(shift_rows), encoding="utf-8")

rows = []
for dom in ["A", "B", "C", "D"]:
    for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                      ("LLM consensus", "llm_consensus")]:
        s = as_num(bloom.loc[bloom.domain == dom, col]).dropna()
        if len(s) == 0:
            continue
        n = len(s)
        k = int((s >= 4).sum())
        lo, hi = wilson_ci(k, n)
        rows.append(f"{dom} & {name} & {k} & {n} & {100.0 * k / n:.1f} "
                    f"[{100.0 * lo:.1f}, {100.0 * hi:.1f}] \\\\")
(TAB / "tab_shift_by_domain.txt").write_text("\n".join(rows), encoding="utf-8")

rows = []
for (stage, grade), g in bloom.groupby(["stage", "grade"]):
    for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                      ("LLM consensus", "llm_consensus")]:
        s = as_num(g[col]).dropna()
        if len(s) == 0:
            continue
        n = len(s)
        k = int((s >= 4).sum())
        lo, hi = wilson_ci(k, n)
        rows.append(f"{stage}{grade} & {name} & {k} & {n} & {100.0 * k / n:.1f} "
                    f"[{100.0 * lo:.1f}, {100.0 * hi:.1f}] \\\\")
(TAB / "tab_shift_by_grade.txt").write_text("\n".join(rows), encoding="utf-8")

model_kappa = []
for m in MODELS:
    for run in base_cols[m]:
        for hum in ["coder1", "coder2"]:
            kb = weighted_kappa(bloom[run], bloom[hum])
            ka = cohen_kappa(abc[run], abc[hum])
            model_kappa.append({"model": m, "run": run, "human": hum,
                                "kappa_bloom_qw": kb, "kappa_abc": ka})
df_mk = pd.DataFrame(model_kappa)
df_mk.to_csv(TAB / "tab_kappa_by_model.csv", index=False)

cons_cols = ["coder1", "coder2"] + [f"{m}_consensus" for m in MODELS]
controversy = []
for _, row in bloom.iterrows():
    vals = [row[c] for c in cons_cols if pd.notna(row[c])]
    if not vals:
        continue
    controversy.append({"stage": row["stage"], "grade": row["grade"],
                        "code": row["code"],
                        "n_distinct": len(set(vals)),
                        "vals": "|".join(str(int(v)) for v in vals)})
df_c = pd.DataFrame(controversy).sort_values(["n_distinct", "code"],
                                              ascending=[False, True]).head(10)
df_c.to_csv(TAB / "tab_top10_controversial.csv", index=False)

infer_rows = []
a = as_num(bloom["coder1"])
b = as_num(bloom["coder2"])
msk = a.notna() & b.notna()
if msk.sum() > 5:
    stat, p = stats.wilcoxon(a[msk], b[msk])
    infer_rows.append({"test": "Wilcoxon_coder1_vs_coder2_bloom",
                       "statistic": round(stat, 3), "p": round(p, 4),
                       "df": int(msk.sum())})
for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                  ("LLM", "llm_consensus")]:
    groups = [as_num(bloom.loc[bloom.domain == d, col]).dropna().astype(int).values
              for d in ["A", "B", "C", "D"]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        infer_rows.append({"test": f"Kruskal_domain_bloom_{name}",
                           "statistic": round(stat, 3), "p": round(p, 4),
                           "df": len(groups) - 1})
pd.DataFrame(infer_rows).to_csv(TAB / "tab_inferential_tests.csv", index=False)


# ================================================================== FIGURES
# --- Fig 1: Bloom distribution
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(6)
w = 0.27
b1 = ax.bar(x - w, c1_b.values, w, label="Coder 1", color=C1, edgecolor=CEDGE, lw=0.6)
b2 = ax.bar(x, c2_b.values, w, label="Coder 2", color=C2, edgecolor=CEDGE, lw=0.6)
b3 = ax.bar(x + w, cl_b.values, w, label="LLM consensus", color=CLLM, edgecolor=CEDGE, lw=0.6)
ymax = max(c1_b.max(), c2_b.max(), cl_b.max()) * 1.18
ax.set_ylim(0, ymax)
ax.axvspan(2.5, 5.5, color=CSHIFT, alpha=0.20, zorder=0)
ax.text(4.0, ymax * 0.97, "shift-sensitive (Bloom ≥ 4)", ha="center", va="top",
        fontsize=9, color="#7A1F1F", style="italic")
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                ha="center", va="bottom", fontsize=8, color=CEDGE)
ax.set_xticks(x, BLOOM_LABELS)
ax.set_ylabel("Number of outcomes")
ax.set_xlabel("Bloom level (revised taxonomy)")
ax.set_title("Distribution of Bloom levels across 127 outcomes")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, edgecolor=CEDGE)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
save(fig, "fig1_bloom_distribution")

# --- Fig 2: ABC distribution
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(6)
w = 0.27
b1 = ax.bar(x - w, c1_a.values, w, label="Coder 1", color=C1, edgecolor=CEDGE, lw=0.6)
b2 = ax.bar(x, c2_a.values, w, label="Coder 2", color=C2, edgecolor=CEDGE, lw=0.6)
b3 = ax.bar(x + w, cl_a.values, w, label="LLM consensus", color=CLLM, edgecolor=CEDGE, lw=0.6)
ymax = max(c1_a.max(), c2_a.max(), cl_a.max()) * 1.18
ax.set_ylim(0, ymax)
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, int(h),
                ha="center", va="bottom", fontsize=8, color=CEDGE)
ax.set_xticks(x, ABC_LABELS, rotation=20, ha="right")
ax.set_ylabel("Number of outcomes")
ax.set_xlabel("ABC learning type")
ax.set_title("Distribution of ABC learning types across 127 outcomes")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, edgecolor=CEDGE)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
save(fig, "fig2_abc_distribution")


# --- helper: heatmap with white grid, fixed cell size, contrast text
def plot_heatmap(ax, data, xlabels, ylabels, title, cmap, vmin=0, vmax=None,
                 fmt="{:.0f}", cbar_label="", fontsize=11):
    if vmax is None:
        vmax = np.nanmax(data)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > vmax * 0.5 else "#222222"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color, fontsize=fontsize, fontweight="bold")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0)
    return im


# --- Fig 3: Bloom x ABC heatmap
fig, ax = plt.subplots(figsize=(8.5, 5.5))
data = crosstab.values.astype(float)
plot_heatmap(ax, data, ABC_LABELS, BLOOM_LABELS,
             f"KABL-ID matrix: Bloom × ABC (LLM consensus, n={int(data.sum())})",
             cmap="YlOrRd", vmin=0, vmax=data.max(),
             cbar_label="Number of outcomes", fontsize=12)
ax.set_xlabel("ABC learning type")
ax.set_ylabel("Bloom level")
fig.tight_layout()
save(fig, "fig3_bloom_x_abc_heatmap")


def conf_matrix(a, b, k=6):
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    return pd.crosstab(df["a"].astype(int), df["b"].astype(int), dropna=False
                       ).reindex(index=range(1, k + 1), columns=range(1, k + 1),
                                 fill_value=0).values


# --- Fig 4: human confusion Bloom
cm = conf_matrix(bloom["coder1"], bloom["coder2"])
fig, ax = plt.subplots(figsize=(7.0, 5.5))
plot_heatmap(ax, cm.astype(float), BLOOM_LABELS, BLOOM_LABELS,
             "Human–human confusion, Bloom (n=127)",
             cmap="Blues", vmin=0, vmax=cm.max(),
             cbar_label="Count", fontsize=11)
ax.set_xlabel("Coder 2")
ax.set_ylabel("Coder 1")
fig.tight_layout()
save(fig, "fig4_confusion_human_bloom")

# --- Fig 5: human confusion ABC
ct_abc = pd.crosstab(abc["coder1"], abc["coder2"]).reindex(
    index=ABC_LABELS, columns=ABC_LABELS, fill_value=0)
fig, ax = plt.subplots(figsize=(7.5, 6.0))
plot_heatmap(ax, ct_abc.values.astype(float), ABC_LABELS, ABC_LABELS,
             "Human–human confusion, ABC (n=127)",
             cmap="Blues", vmin=0, vmax=ct_abc.values.max(),
             cbar_label="Count", fontsize=11)
ax.set_xlabel("Coder 2")
ax.set_ylabel("Coder 1")
fig.tight_layout()
save(fig, "fig5_confusion_human_abc")


# --- Fig 6: LLM vs both humans, three panels, one colorbar per panel
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
cm_c1 = conf_matrix(bloom["llm_consensus"], bloom["coder1"])
cm_c2 = conf_matrix(bloom["llm_consensus"], bloom["coder2"])
cm_diff = cm_c1 - cm_c2
vmax_abs = max(np.abs(cm_diff).max(), 1)
vmax_common = max(cm_c1.max(), cm_c2.max())

plot_heatmap(axes[0], cm_c1.astype(float), BLOOM_LABELS, BLOOM_LABELS,
             "LLM vs Coder 1", cmap="Blues", vmin=0, vmax=vmax_common,
             cbar_label="Count", fontsize=10)
axes[0].set_xlabel("Coder 1")
axes[0].set_ylabel("LLM consensus")

plot_heatmap(axes[1], cm_c2.astype(float), BLOOM_LABELS, BLOOM_LABELS,
             "LLM vs Coder 2", cmap="Blues", vmin=0, vmax=vmax_common,
             cbar_label="Count", fontsize=10)
axes[1].set_xlabel("Coder 2")
axes[1].set_ylabel("LLM consensus")

im = axes[2].imshow(cm_diff, cmap="RdBu_r", aspect="auto",
                    vmin=-vmax_abs, vmax=vmax_abs)
axes[2].set_xticks(range(6), BLOOM_LABELS, rotation=30, ha="right")
axes[2].set_yticks(range(6), BLOOM_LABELS)
axes[2].set_xticks(np.arange(-0.5, 6, 1), minor=True)
axes[2].set_yticks(np.arange(-0.5, 6, 1), minor=True)
axes[2].grid(which="minor", color="white", linewidth=2.0)
axes[2].grid(which="major", visible=False)
axes[2].tick_params(which="both", length=0)
for s in axes[2].spines.values():
    s.set_visible(False)
for i in range(6):
    for j in range(6):
        v = cm_diff[i, j]
        if v != 0:
            axes[2].text(j, i, f"{v:+d}", ha="center", va="center",
                         color="white" if abs(v) > vmax_abs * 0.5 else "#222222",
                         fontsize=10, fontweight="bold")
axes[2].set_title("Difference (C1 − C2)")
axes[2].set_xlabel("Bloom level")
axes[2].set_ylabel("LLM consensus")
cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
cbar.set_label("Count difference", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8, length=0)
fig.tight_layout()
save(fig, "fig6_confusion_llm_vs_human_bloom")


# --- Fig 7 & 8: hybrid strip + box per model, auto-scaled y-axis
def strip_box_plot(df, value_col, title, fname):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    positions = np.arange(1, len(MODELS) + 1)
    data = [df.loc[df.model == m, value_col].dropna().values for m in MODELS]

    all_vals = np.concatenate([d for d in data if len(d) > 0])
    vmin_data = np.nanmin(all_vals)
    vmax_data = np.nanmax(all_vals)
    ymin = max(0.40, np.floor((vmin_data - 0.10) * 20) / 20)
    ymax = min(1.00, np.ceil((vmax_data + 0.05) * 20) / 20)

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor="#E8E2F0", edgecolor=CEDGE, lw=1.0, alpha=0.7),
                    medianprops=dict(color=CEDGE, lw=2.0),
                    whiskerprops=dict(color=CEDGE, lw=1.0),
                    capprops=dict(color=CEDGE, lw=1.2))
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                   s=55, alpha=0.75, color=CMODELS[MODELS[i]],
                   edgecolor=CEDGE, linewidth=0.6, zorder=5)

    if ymin <= 0.8 <= ymax:
        ax.axhline(0.8, color="#7A1F1F", ls="--", lw=1.2,
                   label="almost perfect (0.8)", zorder=1)
    if ymin <= 0.6 <= ymax:
        ax.axhline(0.6, color=C2, ls="--", lw=1.2,
                   label="substantial (0.6)", zorder=1)

    ax.set_xticks(positions, [m.capitalize() for m in MODELS])
    ax.set_ylabel("Cohen's κ")
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + 0.001, 0.05))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=True, edgecolor=CEDGE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True, alpha=0.3)
    fig.tight_layout()
    save(fig, fname)


strip_box_plot(df_mk, "kappa_bloom_qw",
               "Bloom agreement (quadratic-weighted κ) per model × human",
               "fig7_kappa_boxplot_bloom")
strip_box_plot(df_mk, "kappa_abc",
               "ABC agreement (Cohen's κ) per model × human",
               "fig8_kappa_boxplot_abc")


# --- Fig 9: shift by domain
rows9 = []
for dom in ["A", "B", "C", "D"]:
    for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                      ("LLM consensus", "llm_consensus")]:
        s = as_num(bloom.loc[bloom.domain == dom, col]).dropna()
        if len(s) == 0:
            continue
        rows9.append({"domain": dom, "rater": name,
                      "pct": 100.0 * (s >= 4).mean()})
dom_pivot = pd.DataFrame(rows9).pivot(index="domain", columns="rater", values="pct")
fig, ax = plt.subplots(figsize=(8.5, 4.6))
dom_pivot.plot(kind="bar", ax=ax, color=[C1, C2, CLLM], edgecolor=CEDGE, lw=0.6, width=0.75)
ax.set_ylabel("% of outcomes at Bloom ≥ 4")
ax.set_xlabel("Domain (A=e-Society, B=Digital literacy, C=ICT, D=Computational thinking)")
ax.set_title("Shift-sensitive outcomes by curriculum domain")
ax.legend(title="Rater", loc="upper left", bbox_to_anchor=(1.01, 1.0),
          frameon=True, edgecolor=CEDGE)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
save(fig, "fig9_shift_by_domain")

# --- Fig 10: shift by grade
rows10 = []
for (stage, grade), g in bloom.groupby(["stage", "grade"]):
    for name, col in [("coder1", "coder1"), ("coder2", "coder2"),
                      ("LLM consensus", "llm_consensus")]:
        s = as_num(g[col]).dropna()
        if len(s) == 0:
            continue
        rows10.append({"grade": f"{stage}{grade}", "rater": name,
                       "pct": 100.0 * (s >= 4).mean()})
grade_pivot = pd.DataFrame(rows10).pivot(index="grade", columns="rater", values="pct")
fig, ax = plt.subplots(figsize=(10, 4.6))
grade_pivot.plot(kind="bar", ax=ax, color=[C1, C2, CLLM], edgecolor=CEDGE, lw=0.6, width=0.8)
ax.set_ylabel("% of outcomes at Bloom ≥ 4")
ax.set_xlabel("Grade (OS = primary, SS = secondary)")
ax.set_title("Shift-sensitive outcomes by grade")
ax.legend(title="Rater", loc="upper left", bbox_to_anchor=(1.01, 1.0),
          frameon=True, edgecolor=CEDGE)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
save(fig, "fig10_shift_by_grade")


# --- Fig 11: intra/cross-model heatmap (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (scale, getval, title) in zip(axes, [
        ("Bloom", lambda a, b: weighted_kappa(bloom[a], bloom[b]), "Bloom (quadratic κ)"),
        ("ABC", lambda a, b: cohen_kappa(abc[a], abc[b]), "ABC (Cohen's κ)")]):
    M = np.full((5, 5), np.nan)
    for i, m in enumerate(MODELS):
        vals = [getval(a, b) for a, b in itertools.combinations(base_cols[m], 2)]
        M[i, i] = np.nanmean(vals) if vals else np.nan
    cons_cols_m = [f"{m}_consensus" for m in MODELS]
    for i, j in itertools.combinations(range(5), 2):
        k = getval(cons_cols_m[i], cons_cols_m[j])
        M[i, j] = M[j, i] = k
    plot_heatmap(ax, M, [m.capitalize() for m in MODELS],
                 [m.capitalize() for m in MODELS],
                 f"Intra / cross-model: {title}",
                 cmap="YlGnBu", vmin=0.5, vmax=1.0,
                 fmt="{:.2f}", cbar_label="κ", fontsize=10)
fig.tight_layout()
save(fig, "fig11_intra_model_heatmap")


# --- Fig 12: Pairwise agreement heatmap between all 15 base runs (Bloom)
# 15 runs = 5 models × 3 runs. Matrix 15×15, colored by quadratic-weighted κ.
# Diagonal blocks = intra-model stability; off-diagonal = cross-model consistency.
run_labels = []
run_labels_short = []
for m in MODELS:
    for i, run in enumerate(base_cols[m]):
        run_labels.append(run)
        run_labels_short.append(f"{m[:4]}_{i+1}")

n_runs = len(run_labels)
K = np.full((n_runs, n_runs), np.nan)
for i in range(n_runs):
    for j in range(i, n_runs):
        if i == j:
            K[i, j] = 1.0
            continue
        kij = weighted_kappa(bloom[run_labels[i]], bloom[run_labels[j]])
        K[i, j] = K[j, i] = kij

# plot
fig, ax = plt.subplots(figsize=(9.5, 8.0))
cmap = plt.cm.get_cmap("YlGnBu").copy() if hasattr(plt.cm, "get_cmap") else plt.cm.YlGnBu
im = ax.imshow(K, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="equal")

# ticks
ax.set_xticks(range(n_runs), run_labels_short, rotation=90, fontsize=7)
ax.set_yticks(range(n_runs), run_labels_short, fontsize=7)

# white grid every cell + thick lines between model blocks
ax.set_xticks(np.arange(-0.5, n_runs, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_runs, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.grid(which="major", visible=False)
ax.tick_params(which="both", length=0)

# thick separators between model blocks (every 3 runs)
for k in range(3, n_runs, 3):
    ax.axhline(k - 0.5, color="#222222", lw=1.8)
    ax.axvline(k - 0.5, color="#222222", lw=1.8)

# annotate: only diagonal blocks (values inside are redundant, but readable)
for i in range(n_runs):
    for j in range(n_runs):
        v = K[i, j]
        if np.isnan(v):
            continue
        color = "white" if v > 0.85 else "#222222"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=5.5, color=color)

# spine off
for s in ax.spines.values():
    s.set_visible(False)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Quadratic-weighted κ (Bloom)", fontsize=9)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8, length=0)

ax.set_title("Pairwise agreement between all 15 LLM base runs (Bloom)\n"
             "Blocks separated by model; diagonal = 1 (self-agreement)",
             fontsize=11)
fig.tight_layout()
save(fig, "fig12_bland_altman")


# ================================================================== summary
summary = {
    "n_outcomes": len(bloom),
    "shift_sensitive": {
        "coder1": float((as_num(bloom["coder1"]) >= 4).mean() * 100),
        "coder2": float((as_num(bloom["coder2"]) >= 4).mean() * 100),
        "llm_consensus": float((as_num(bloom["llm_consensus"]) >= 4).mean() * 100),
    },
    "top10_controversial_codes": df_c["code"].tolist(),
}
(ROOT / "results" / "extra_stats_summary.json").write_text(json.dumps(summary, indent=2))

print("\nSummary:")
print(json.dumps(summary, indent=2))
print(f"\nAll tables -> {TAB}")
print(f"All figures -> {FIG}")