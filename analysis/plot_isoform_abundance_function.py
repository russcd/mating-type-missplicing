#!/usr/bin/env python3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# -----------------------
# Argparse for input/output
# -----------------------
parser = argparse.ArgumentParser(description="Two-panel isoform analysis (side-by-side layout)")
parser.add_argument("--input", required=True, help="Input TSV file with isoform info")
parser.add_argument("--output", default="isoforms_two_panel.pdf", help="Output PDF plot")
args = parser.parse_args()

# -----------------------
# Set plotting context
# -----------------------
sns.set_context("paper", font_scale=0.9)

# -----------------------
# Load the data
# -----------------------
df = pd.read_csv(args.input, sep=r'\s+', engine='python')

# Convert numeric columns
numeric_cols = ["Introns", "Expression", "Num_Isoforms", "Start", "End", "Functional"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)

# Log-transform expression for modeling
df["Expression_log"] = np.log1p(df["Expression"])

# Cap introns at 5
df["Introns_capped"] = df["Introns"].apply(lambda x: x if x <= 5 else 5)

# -----------------------
# Fit OLS model (kept for reporting)
# -----------------------
X = sm.add_constant(df[["Expression_log", "MatingTypeRegion", "Introns_capped"]])
y = df["Num_Isoforms"]
model = sm.OLS(y, X).fit()
print(model.summary())

mean_expr = df["Expression_log"].mean()
mean_introns = df["Introns_capped"].mean()
X_pred = pd.DataFrame({
    "const": 1,
    "Expression_log": [mean_expr, mean_expr],
    "MatingTypeRegion": [0, 1],
    "Introns_capped": [mean_introns, mean_introns]
})
pred_isoforms = model.predict(X_pred)
print("\nPredicted Num_Isoforms at mean expression and mean intron number:")
print(pd.DataFrame({"MatingTypeRegion": [0, 1], "Num_Isoforms_pred": pred_isoforms}))

# -----------------------
# Compute proportion functional
# -----------------------
df = df[df["Num_Isoforms"] > 0].copy()
df["PropFunctional"] = df["Functional"] / df["Expression"]

# -----------------------
# Plotting
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

# ---- Panel 1: Isoforms vs Introns ----
sns.boxplot(
    data=df,
    x="Introns_capped",
    y="Num_Isoforms",
    hue="MatingTypeRegion",
    palette={0: "steelblue", 1: "orange"},
    showcaps=True,
    boxprops={'facecolor': 'None'},
    showfliers=False,
    whiskerprops={'linewidth': 1.2},
    width=2/3,
    ax=ax1
)
sns.stripplot(
    data=df,
    x="Introns_capped",
    y="Num_Isoforms",
    hue="MatingTypeRegion",
    palette={0: "steelblue", 1: "orange"},
    dodge=True,
    alpha=0.5,
    size=3,
    ax=ax1
)
ax1.set_ylabel("Number of Isoforms", fontsize=9)
ax1.set_xlabel("Number of Introns", fontsize=9)
ax1.get_legend().remove()

# ---- Panel 2: Proportion functional isoforms vs Introns ----
sns.boxplot(
    data=df,
    x="Introns_capped",
    y="PropFunctional",
    hue="MatingTypeRegion",
    palette={0: "steelblue", 1: "orange"},
    showcaps=True,
    showfliers=False,
    boxprops={'facecolor': 'None'},
    whiskerprops={'linewidth': 1.2},
    width=2/3,
    ax=ax2
)
sns.stripplot(
    data=df,
    x="Introns_capped",
    y="PropFunctional",
    hue="MatingTypeRegion",
    palette={0: "steelblue", 1: "orange"},
    dodge=True,
    alpha=0.5,
    size=3,
    ax=ax2
)
ax2.set_ylabel("Proportion Transcripts > 80% CDS Length", fontsize=9)
ax2.set_xlabel("Number of Introns", fontsize=9)

xticks = ax2.get_xticks()
ax2.set_xticklabels([str(int(t)) if t < 5 else "5+" for t in xticks], fontsize=8)

# ---- Shared legend directly above plots ----
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(),
    ["Autosomes", "Mating-Type Region"],
    title="",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),  # right above the axes
    ncol=2,
    fontsize=8,
    title_fontsize=9
)
ax2.get_legend().remove()

# Adjust spacing so legend is snug
plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.3)
plt.savefig(args.output)
plt.close()
