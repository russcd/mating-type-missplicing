#!/usr/bin/env python
"""Within-gene test: does perc_A predict IR among MT-region isoforms?
"""
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IR_SUBCATEGORIES = {"intron_retention", "mono-exon_by_intron_retention"}


def parse_gtf_attrs(s):
    out = {}
    for kv in s.strip().rstrip(";").split(";"):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition(" ")
        out[k] = v.strip().strip('"')
    return out


def load_isoform_spans(gtf_path):
    """Return (spans, gene_of).

    spans: dict isoform_id -> (chrom, start, end) using transcript outer span
    gene_of: dict isoform_id -> gene_id
    """
    exons = defaultdict(list)
    chrom_of = {}
    gene_of = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attrs = parse_gtf_attrs(fields[8])
            iid = attrs.get("transcript_id")
            if not iid:
                continue
            if fields[2] == "transcript":
                gid = attrs.get("gene_id")
                if gid:
                    gene_of[iid] = gid
            elif fields[2] == "exon":
                exons[iid].append((int(fields[3]), int(fields[4])))
                chrom_of[iid] = fields[0]
    spans = {}
    for iid, ex in exons.items():
        ex.sort()
        spans[iid] = (chrom_of[iid], ex[0][0], ex[-1][1])
    return spans, gene_of


def load_sqanti(path):
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "isoform",
            "chrom",
            "subcategory",
            "perc_A_downstream_TTS",
            "RTS_stage",
        ],
        dtype={"isoform": str, "chrom": str, "subcategory": str},
    )
    df["perc_A_downstream_TTS"] = pd.to_numeric(
        df["perc_A_downstream_TTS"], errors="coerce"
    )
    return df


def fit_logistic(X, y, term_names):
    """Fit unregularized logistic regression with k predictors and return per-term
    Wald statistics plus model fit metrics.

    X: (n, k) array (no intercept column; we add it for the Fisher info matrix)
    y: (n,) binary
    term_names: list of length k naming the predictor columns
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=int)

    model = LogisticRegression(C=1e10, max_iter=2000, solver="lbfgs")
    model.fit(X, y)
    intercept = float(model.intercept_[0])
    coefs = model.coef_[0].astype(float)

    # Wald SE/p from inverse Fisher information at the MLE
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    p_hat = model.predict_proba(X)[:, 1]
    W = p_hat * (1 - p_hat)
    fisher = X_design.T @ (W[:, None] * X_design)
    try:
        cov = np.linalg.inv(fisher)
        ses = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        ses = np.full(X_design.shape[1], np.nan)

    rows = []
    rows.append(
        {
            "term": "(intercept)",
            "estimate": intercept,
            "std_err": float(ses[0]),
            "z": intercept / ses[0] if ses[0] > 0 else float("nan"),
            "p_value": (
                2 * (1 - norm.cdf(abs(intercept / ses[0]))) if ses[0] > 0 else float("nan")
            ),
            "odds_ratio": float("nan"),
        }
    )
    for i, name in enumerate(term_names):
        b = float(coefs[i])
        se = float(ses[i + 1])
        z = b / se if se > 0 else float("nan")
        pval = 2 * (1 - norm.cdf(abs(z))) if not math.isnan(z) else float("nan")
        rows.append(
            {
                "term": name,
                "estimate": b,
                "std_err": se,
                "z": z,
                "p_value": pval,
                "odds_ratio": float(math.exp(b)),
            }
        )

    proba = model.predict_proba(X)[:, 1]
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    eps = 1e-12
    ll_full = float(np.sum(y * np.log(proba + eps) + (1 - y) * np.log(1 - proba + eps)))
    p0 = float(np.mean(y))
    ll_null = float(np.sum(y * np.log(p0 + eps) + (1 - y) * np.log(1 - p0 + eps)))
    pseudo_r2 = 1 - ll_full / ll_null if ll_null != 0 else float("nan")

    return {
        "terms": rows,
        "auc": auc,
        "pseudo_r2": pseudo_r2,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def fmt(x, spec="+.5f"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return format(x, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sqanti", required=True)
    p.add_argument("--isoform-gtf", required=True)
    p.add_argument("--mt-chrom", default="scaffold_2")
    p.add_argument("--mt-start", type=int, default=45993)
    p.add_argument("--mt-end", type=int, default=1729827)
    p.add_argument("--out-tsv", required=True, help="Per-isoform TSV")
    p.add_argument("--summary-tsv", required=True, help="Per-term model summary TSV")
    p.add_argument(
        "--out-plot-base",
        default=None,
        help="If set, writes a per-gene violin plot to {base}.pdf and {base}.png",
    )
    args = p.parse_args()

    print("Loading SQANTI3 classification...")
    sq = load_sqanti(args.sqanti)
    print(f"  {len(sq)} isoforms total")

    n_rts = int((sq["RTS_stage"] == True).sum())
    sq = sq[sq["RTS_stage"] != True].reset_index(drop=True)
    print(f"  Dropped {n_rts} RT-switching artifacts (RTS_stage=True); {len(sq)} remain")

    print("Loading isoform GTF spans...")
    spans, gene_of = load_isoform_spans(args.isoform_gtf)
    print(f"  {len(spans)} isoform spans, {len(gene_of)} gene assignments")

    span_df = pd.DataFrame(
        [(iid, s, e) for iid, (_, s, e) in spans.items()],
        columns=["isoform", "start", "end"],
    )
    df = sq.merge(span_df, on="isoform", how="inner")
    if len(df) != len(sq):
        print(f"WARNING: dropped {len(sq) - len(df)} isoforms with no GTF span")

    df["has_IR"] = df["subcategory"].isin(IR_SUBCATEGORIES)
    df["is_MT"] = (
        (df["chrom"] == args.mt_chrom)
        & (df["start"] >= args.mt_start)
        & (df["end"] <= args.mt_end)
    ).astype(int)
    df["gene_id"] = df["isoform"].map(gene_of)
    df = df.dropna(subset=["perc_A_downstream_TTS"]).reset_index(drop=True)

    print(
        f"  IR={int(df['has_IR'].sum())}  non-IR={int((~df['has_IR']).sum())}  "
        f"MT={int(df['is_MT'].sum())}  AU={int((df['is_MT']==0).sum())}"
    )

    df[
        ["isoform", "chrom", "start", "end", "gene_id", "is_MT", "has_IR",
         "perc_A_downstream_TTS"]
    ].to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote per-isoform TSV: {args.out_tsv}")

    # MT-only + gene fixed effects: controls for gene-level IR propensity, so
    # the perc_A coefficient reflects purely within-gene variation.
    mt_sub = df[df["is_MT"] == 1].reset_index(drop=True)
    mt_gene = mt_sub.dropna(subset=["gene_id"]).copy()
    gene_var = mt_gene.groupby("gene_id")["has_IR"].agg(["sum", "count"])
    gene_var["n_non_ir"] = gene_var["count"] - gene_var["sum"]
    informative_genes = gene_var[
        (gene_var["sum"] > 0) & (gene_var["n_non_ir"] > 0)
    ].index
    mt_gene_inf = mt_gene[mt_gene["gene_id"].isin(informative_genes)].reset_index(
        drop=True
    )
    n_all_ir = int((gene_var["n_non_ir"] == 0).sum())
    n_no_ir = int((gene_var["sum"] == 0).sum())
    print(
        f"\nMT gene-FE subset: {len(mt_gene_inf)} isoforms across "
        f"{len(informative_genes)} genes with IR variation "
        f"(dropped {n_all_ir} all-IR + {n_no_ir} all-non-IR genes)"
    )
    gene_dummies = pd.get_dummies(
        mt_gene_inf["gene_id"], drop_first=True, dtype=float
    )
    X_gene = np.hstack(
        [mt_gene_inf[["perc_A_downstream_TTS"]].values, gene_dummies.values]
    )
    term_names_gene = ["perc_A_downstream_TTS"] + [
        f"gene:{c}" for c in gene_dummies.columns
    ]
    fit_mt_gene = fit_logistic(
        X_gene,
        mt_gene_inf["has_IR"].astype(int).values,
        term_names_gene,
    )

    # Print only perc_A + intercept (not 163 gene dummies)
    print(
        f"\n--- MT + gene FE:  P(IR) ~ perc_A + gene_id  (MT, gene fixed effects) ---"
    )
    print(
        f"  n={fit_mt_gene['n']}  n_IR={fit_mt_gene['n_pos']}  "
        f"n_genes={len(informative_genes)}  "
        f"AUC={fit_mt_gene['auc']:.4f}  pseudo_R2={fit_mt_gene['pseudo_r2']:.4f}"
    )
    print(
        f"  {'term':<22s} {'estimate':>12s} {'std_err':>10s} "
        f"{'z':>8s} {'p':>11s} {'OR':>10s}"
    )
    for t in fit_mt_gene["terms"][:2]:  # intercept + perc_A only
        print(
            f"  {t['term']:<22s} {fmt(t['estimate'],'+.5f'):>12s} "
            f"{fmt(t['std_err'],'.5f'):>10s} {fmt(t['z'],'+.2f'):>8s} "
            f"{fmt(t['p_value'],'.2e'):>11s} {fmt(t['odds_ratio'],'.3f'):>10s}"
        )
    perc_a_term = fit_mt_gene["terms"][1]
    or_10 = math.exp(10 * perc_a_term["estimate"])
    print(f"  OR per +10 pp: {or_10:.3f}")

    # Summary TSV: intercept + perc_A term from the gene-FE model
    rows = []
    for t in fit_mt_gene["terms"][:2]:
        rows.append(
            {
                "model": "mt_only_gene_fe",
                "term": t["term"],
                "estimate": t["estimate"],
                "std_err": t["std_err"],
                "z": t["z"],
                "p_value": t["p_value"],
                "odds_ratio": t["odds_ratio"],
                "model_n": fit_mt_gene["n"],
                "model_n_IR": fit_mt_gene["n_pos"],
                "model_auc": fit_mt_gene["auc"],
                "model_pseudo_r2": fit_mt_gene["pseudo_r2"],
            }
        )
    pd.DataFrame(rows).to_csv(args.summary_tsv, sep="\t", index=False)
    print(f"\nWrote summary TSV: {args.summary_tsv}")

    if args.out_plot_base:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Per-gene mean perc_A for IR isoforms and non-IR isoforms, one paired
        # observation per gene with IR variation.
        ir_means = []
        nir_means = []
        for gid, g in mt_gene_inf.groupby("gene_id"):
            ir_means.append(
                g.loc[g["has_IR"], "perc_A_downstream_TTS"].mean()
            )
            nir_means.append(
                g.loc[~g["has_IR"], "perc_A_downstream_TTS"].mean()
            )
        ir_means = np.array(ir_means)
        nir_means = np.array(nir_means)
        n_genes = len(ir_means)

        fig, ax = plt.subplots(figsize=(6, 5))
        color_nir = "#4477AA"
        color_ir = "#FF8C00"

        parts = ax.violinplot(
            [nir_means, ir_means],
            positions=[1, 2],
            widths=0.7,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, color in zip(parts["bodies"], [color_nir, color_ir]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.6)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [
                f"non-IR isoforms\n(n={n_genes} genes)",
                f"IR isoforms\n(n={n_genes} genes)",
            ]
        )
        ax.set_ylabel("Mean % A downstream of TTS (per gene)")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_path = f"{args.out_plot_base}.{ext}"
            fig.savefig(out_path, dpi=150)
            print(f"Wrote plot: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
