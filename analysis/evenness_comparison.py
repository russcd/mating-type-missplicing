#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Test if Evenness differs between MatingTypeRegion vs non-region after correcting for expression and introns.")
    parser.add_argument("input", help="Input TSV file")
    parser.add_argument("output_prefix", help="Prefix for output files")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, sep="\t")

    # Log-transform expression
    df["log_expr"] = np.log1p(df["Expression"])

    # Base model: Evenness ~ log_expr + Introns
    model_base = smf.ols("Evenness ~ log_expr + Introns", data=df).fit()

    # Full model: Evenness ~ log_expr + Introns + MatingTypeRegion
    model_full = smf.ols("Evenness ~ log_expr + Introns + MatingTypeRegion", data=df).fit()

    # Save summaries
    with open(f"{args.output_prefix}_model_base.txt", "w") as f:
        f.write(model_base.summary().as_text())

    with open(f"{args.output_prefix}_model_full.txt", "w") as f:
        f.write(model_full.summary().as_text())

    # Likelihood ratio test (ANOVA comparison)
    anova_res = anova_lm(model_base, model_full)
    anova_res.to_csv(f"{args.output_prefix}_anova.csv")

    # Print quick results
    print("=== Base model ===")
    print(model_base.summary())
    print("\n=== Full model ===")
    print(model_full.summary())
    print("\n=== Model comparison (likelihood ratio test) ===")
    print(anova_res)

    # Compute residuals from base model to visualize added explanatory power
    df["resid_base"] = model_base.resid

    # Plot residuals vs MatingTypeRegion
    plt.figure(figsize=(5,4))
    df.boxplot(column="resid_base", by="MatingTypeRegion")
    plt.title("Residual Evenness after correcting for Expression and Introns")
    plt.suptitle("")
    plt.xlabel("MatingTypeRegion (0=Outside, 1=Inside)")
    plt.ylabel("Residual Evenness")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_residual_plot.pdf")

if __name__ == "__main__":
    main()
