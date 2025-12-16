#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import partial_dependence
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from kneed import KneeLocator

def clean_feature_name(feat):
    """Convert feature names to publication-quality labels."""
    replacements = {
        "GC_content": "GC Fraction",
        "distance_from_gene_start": "Distance from TSS (bp)",
        "TPM": "TPM (log scale)",
        "Intron_length": "Intron Length (bp)",
        "Frame_disruption": "Frame Disruption",
        "First2_canonical": "5' Splice Site Canonical",
        "Last2_canonical": "3' Splice Site Canonical"
    }
    
    # Check for exact matches first
    if feat in replacements:
        return replacements[feat]
    
    # Handle context features
    if "FivePrimeContext" in feat or "ThreePrimeContext" in feat:
        parts = feat.split("_")
        if "FivePrimeContext" in feat:
            pos = parts[1]
            base = parts[2]
            return f"5' pos {pos} = {base}"
        elif "ThreePrimeContext" in feat:
            pos = parts[1]
            base = parts[2]
            return f"3' pos {pos} = {base}"
    
    # Default: replace underscores with spaces and title case
    return feat.replace("_", " ").title()

def featurize_sequence_context(df, five_col="FivePrimeContext", three_col="ThreePrimeContext"):
    """One-hot encode nucleotides at each position, skipping canonical splice site positions."""
    bases = ["A", "C", "G", "T", "N"]

    if five_col in df.columns:
        seq_len = df[five_col].str.len().max()
        for i in range(seq_len):
            if i in [0,1]:  # skip first 2 positions (canonical 5' site)
                continue
            for base in bases:
                df[f"{five_col}_{i+1}_{base}"] = (df[five_col].str[i].fillna("N") == base).astype(int)

    if three_col in df.columns:
        seq_len = df[three_col].str.len().max()
        for i in range(seq_len):
            if i in [-2,-1]:  # skip last 2 positions (canonical 3' site)
                continue
            for base in bases:
                df[f"{three_col}_{i+1}_{base}"] = (df[three_col].str[i].fillna("N") == base).astype(int)

    return df

def train_and_eval_without_feature(feat, X_train, X_test, y_train, y_test, rf_params, baseline_auc):
    """Helper for parallel LOFO (Leave-One-Feature-Out) analysis."""
    X_train_reduced = X_train.drop(columns=[feat], errors="ignore")
    X_test_reduced = X_test.drop(columns=[feat], errors="ignore")

    rf_params_local = rf_params.copy()
    rf_params_local.pop("random_state", None)
    rf_params_local.pop("n_jobs", None)
    rf_params_local.pop("class_weight", None)

    model = RandomForestClassifier(**rf_params_local, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train_reduced, y_train)
    auc_reduced = roc_auc_score(y_test, model.predict_proba(X_test_reduced)[:, 1])
    return {"Feature": feat, "AUC_without": auc_reduced, "AUC_drop": baseline_auc - auc_reduced}

def main():
    parser = argparse.ArgumentParser(description="Random Forest classification with automated minimal model selection, LOFO, PDPs, and saturation curve")
    parser.add_argument("--input", required=True, help="Input TSV with features")
    parser.add_argument("--output", default="output", help="Base filename for all outputs")
    parser.add_argument("--threshold", type=float, default=0.025, help="Retention ratio threshold for misspliced introns")
    args = parser.parse_args()
    
    # Set up output filenames
    output_base = args.output
    importance_file = f"{output_base}_importances.tsv"
    elbow_plot = f"{output_base}_feature_auc_elbow_curve.pdf"
    scatter_plot = f"{output_base}_feature_auc_scatter.pdf"
    pdp_plot = f"{output_base}_partial_dependence_plots.pdf"

    # === Load data ===
    df = pd.read_csv(args.input, sep="\t")
    if "Expressed" in df.columns and "In_region" in df.columns:
        df = df[(df["Expressed"] == 1) & (df["In_region"] == 1)]
        print(f"{len(df)} expressed introns retained for modeling")

    if "gene_id" not in df.columns:
        raise ValueError("gene_id column is required for gene-level splitting.")

    # === Binary target ===
    y = (df["rMATS_retention_ratio"] >= args.threshold).astype(int)

    # Canonical splice sites
    if "First2" in df.columns:
        df["First2_canonical"] = df["First2"].isin(["GT","GC"]).astype(int)
    if "Last2" in df.columns:
        df["Last2_canonical"] = (df["Last2"]=="AG").astype(int)

    # Frame disruption
    df["Frame_disruption"] = 1 - df.get("Frame_preserving", 0)

    # Sequence context
    df = featurize_sequence_context(df)

    # Drop unneeded columns
    drop_cols = ["rMATS_retention_ratio","Header","First2","Last2","Frame_preserving",
                 "Expressed","Average_gene_depth","Gene_length","In_region",
                 "FivePrimeContext","ThreePrimeContext","gene_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "TPM" not in X.columns:
        raise ValueError("TPM column is required in the input file.")

    gene_ids = df["gene_id"].copy()
    X = pd.get_dummies(X)
    print(f"Using {X.shape[1]} features for modeling")

    # === Gene-level train/test split ===
    unique_genes = gene_ids.unique()
    train_genes, test_genes = train_test_split(unique_genes, test_size=0.2, random_state=42)
    train_mask = gene_ids.isin(train_genes)
    test_mask = gene_ids.isin(test_genes)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    gene_ids_train = gene_ids[train_mask]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Class balance train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # === Hyperparameter tuning ===
    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    param_dist = {
        "n_estimators": randint(100,400),
        "max_depth": randint(10,30),
        "min_samples_split": randint(2,10),
        "min_samples_leaf": randint(1,5),
        "max_features": ["sqrt","log2",None]
    }
    search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=30,
        cv=GroupKFold(n_splits=3), scoring="roc_auc",
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train, groups=gene_ids_train)
    best_model = search.best_estimator_
    print(f"Best params: {search.best_params_}")

    # === Baseline AUC ===
    baseline_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
    print(f"Baseline ROC-AUC: {baseline_auc:.3f}")

    # === Feature importances from full model (for ranking) ===
    importances = best_model.feature_importances_
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)

    # === Prepare RF params ===
    rf_params = best_model.get_params().copy()
    rf_params.pop("random_state", None)
    rf_params.pop("n_jobs", None)
    rf_params.pop("class_weight", None)

    # === Saturation curve computation ===
    print("Computing feature saturation curve...")
    feature_aucs = []
    num_features_list = []
    sorted_feats = feature_importances["Feature"].tolist()

    for i in tqdm(range(1, len(sorted_feats) + 1)):
        feats = sorted_feats[:i]
        model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1, class_weight="balanced")
        model.fit(X_train[feats], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test[feats])[:, 1])
        feature_aucs.append(auc)
        num_features_list.append(i)

    # === Minimal model selection via elbow (with fallback) ===
    print("Selecting minimal model using elbow detection...")
    kn = KneeLocator(num_features_list, feature_aucs, curve='concave', direction='increasing')

    if kn.knee is not None:
        optimal_n = int(kn.knee)
        method = "elbow"
        print(f"Elbow detected at {optimal_n} features.")
    else:
        # fallback to 99% of baseline
        retained_auc = 0.99 * baseline_auc
        optimal_n = next((i for i, auc in enumerate(feature_aucs, start=1) if auc >= retained_auc), len(sorted_feats))
        method = "99% baseline"
        print(f"No clear elbow detected — falling back to 99% baseline rule ({optimal_n} features).")

    selected_features = sorted_feats[:optimal_n]
    current_auc = feature_aucs[optimal_n - 1]
    print(f"Minimal model selected {optimal_n} features (AUC={current_auc:.3f}) using {method} method.")

    # === Train final model and get importances ===
    print("Training final model with selected features...")
    final_model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1, class_weight="balanced")
    final_model.fit(X_train[selected_features], y_train)
    
    # Get importances from final model
    final_importances = final_model.feature_importances_
    final_feature_importances = pd.DataFrame({
        "Feature": selected_features, 
        "Importance": final_importances
    }).sort_values("Importance", ascending=False)
    final_feature_importances.to_csv(importance_file, sep="\t", index=False)
    print(f"Final model feature importances written to {importance_file}")

    # visualize elbow curve
    plt.figure(figsize=(6, 4))
    plt.plot(num_features_list, feature_aucs, marker='o', label='Cumulative AUC')
    if kn.knee is not None:
        plt.axvline(optimal_n, color='red', linestyle='--', label=f'Elbow = {optimal_n} features')
    else:
        plt.axhline(0.99 * baseline_auc, color='red', linestyle='--', label='99% AUC threshold')
    plt.axhline(baseline_auc, color='gray', linestyle='--', label='Full model AUC')
    plt.xlabel("Number of Features Included")
    plt.ylabel("ROC-AUC")
    plt.title("Feature Saturation Curve with Elbow Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(elbow_plot)
    plt.close()
    print(f"Elbow curve saved to {elbow_plot}")

    # === LOFO ===
    lofo_results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                train_and_eval_without_feature,
                feat,
                X_train[selected_features],
                X_test[selected_features],
                y_train,
                y_test,
                rf_params,
                current_auc
            ): feat
            for feat in selected_features
        }
        for f in tqdm(as_completed(futures), total=len(futures)):
            lofo_results.append(f.result())
    lofo_df = pd.DataFrame(lofo_results)

    # === Single-feature AUC ===
    single_results = []
    for feat in tqdm(selected_features):
        model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1, class_weight="balanced")
        model.fit(X_train[[feat]], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test[[feat]])[:, 1])
        single_results.append({"Feature": feat, "Single_AUC": auc})
    single_df = pd.DataFrame(single_results)

    # === Merge results for scatterplot ===
    merged = pd.merge(single_df, lofo_df, on="Feature")
    merged = pd.merge(merged, final_feature_importances, on="Feature")

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        merged["Single_AUC"],
        merged["AUC_drop"],
        c=merged["Importance"],
        cmap="viridis",
        s=80,
        edgecolor="k"
    )
    plt.colorbar(scatter, label="Feature Importance")
    for _, row in merged.iterrows():
        plt.text(row["Single_AUC"], row["AUC_drop"], row["Feature"], fontsize=8, ha="right", va="bottom")
    plt.xlabel("Single-feature ROC-AUC")
    plt.ylabel("AUC Drop (Leave-One-Feature-Out)")
    plt.title("Feature Comparison: Single AUC vs LOFO AUC Drop")
    plt.tight_layout()
    plt.savefig(scatter_plot)
    plt.close()
    print(f"Scatterplot saved to {scatter_plot}")

    # === Partial Dependence Plots ===
    print("Generating partial dependence plots...")
    # Use the final_model already trained above

    # Calculate number of rows needed (max 4 plots per row)
    n_features = len(selected_features)
    n_cols = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        
        # Get feature index in the selected features
        feat_idx = list(X_train[selected_features].columns).index(feat)
        
        # Calculate partial dependence
        pd_result = partial_dependence(
            final_model, 
            X_train[selected_features], 
            [feat_idx],
            grid_resolution=50
        )
        
        # Plot as a line
        ax.plot(pd_result['grid_values'][0], pd_result['average'][0], 'b-', linewidth=2)
        
        # Clean up the feature name for x-axis
        clean_name = clean_feature_name(feat)
        ax.set_xlabel(clean_name, fontsize=10)
        
        # Only label y-axis on leftmost plots
        if idx % n_cols == 0:
            ax.set_ylabel("Partial Dependence", fontsize=10)
        else:
            ax.set_ylabel("")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide any unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(pdp_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Partial dependence plots saved to {pdp_plot}")

if __name__ == "__main__":
    main()
