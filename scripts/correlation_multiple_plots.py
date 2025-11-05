import pandas as pd
import glob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load and process experiments ---
folder = "../metrics_results"
files = glob.glob(f"{folder}/*.csv")

# Helper: detect clustering algorithm from experiment name
def get_algorithm(name):
    name = name.lower()
    if "k_means" in name or "km" in name:
        return "k_means"
    elif "hdbscan" in name:
        return "hdbscan"
    elif "agglomerative" in name or name.startswith("ac_") or "_ac_" in name:
        return "agglomerative"
    else:
        return "other"

# Helper: detect UMAP and normalization
def get_umap_norm(name):
    name = name.lower()
    umap = "UMAP" if "umap" in name else "No UMAP"
    norm = "Normalised" if "normalised" in name else "Not normalised"
    return pd.Series([umap, norm])

# --- Collect global AMI & silhouette scores ---
data = []
for file in files:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if not {"type", "silhouette_score", "ami_score"}.issubset(df.columns):
        print(f"⚠️ Skipping {file}: missing required columns.")
        continue

    global_row = df[df["type"].str.lower() == "global"]
    if global_row.empty:
        print(f"⚠️ No 'global' row found in {file}, skipping.")
        continue

    ami = pd.to_numeric(global_row["ami_score"].values[0], errors="coerce")
    sil = pd.to_numeric(global_row["silhouette_score"].values[0], errors="coerce")

    if pd.notna(ami) and pd.notna(sil):
        experiment_name = file.split("/")[-1].replace(".csv", "")
        algo = get_algorithm(experiment_name)
        umap, norm = get_umap_norm(experiment_name)
        data.append({
            "experiment": experiment_name,
            "algorithm": algo,
            "UMAP": umap,
            "Normalised": norm,
            "AMI_score": ami,
            "silhouette_score": sil
        })

df_all = pd.DataFrame(data)
if df_all.empty:
    raise ValueError("No valid data found! Check that your CSVs have 'global' rows with AMI and silhouette scores.")

# --- Compute Pearson correlations per (algorithm, normalisation, UMAP) ---
results = []
for (algo, norm, umap), group in df_all.groupby(["algorithm", "Normalised", "UMAP"]):
    if len(group) < 3:
        continue
    pearson_corr, pearson_p = pearsonr(group["AMI_score"], group["silhouette_score"])
    pearson_sig = "Significant" if pearson_p < 0.05 else "Not significant"
    results.append({
        "algorithm": algo,
        "Normalised": norm,
        "UMAP": umap,
        "r": pearson_corr,
        "p_value": pearson_p,
        "significance": pearson_sig
    })

summary_df = pd.DataFrame(results)
summary_df.to_csv("ami_silhouette_pearson_by_algo_norm_umap.csv", index=False)
print("\n=== Pearson Correlation Results ===")
print(summary_df)

# --- Visualization setup ---
sns.set(style="whitegrid", font_scale=1.1)

algorithms = sorted(df_all["algorithm"].unique())
norm_states = ["Not normalised", "Normalised"]
umap_states = ["No UMAP", "UMAP"]

# Color palette per algorithm
algo_colors = {
    "k_means": "#E76F51",
    "agglomerative": "#2A9D8F",
    "hdbscan": "#264653",
    "other": "#999999"
}

# --- Plot grid for each UMAP state ---
for umap_state in umap_states:
    subset_umap = df_all[df_all["UMAP"] == umap_state]
    if subset_umap.empty:
        continue

    fig, axes = plt.subplots(
        nrows=len(algorithms), ncols=len(norm_states),
        figsize=(12, 4 * len(algorithms)),
        sharex=True, sharey=True
    )

    if len(algorithms) == 1:
        axes = np.array([axes])  # Ensure 2D array
    axes = np.atleast_2d(axes)

    for i, algo in enumerate(algorithms):
        for j, norm_state in enumerate(norm_states):
            ax = axes[i, j]
            subset = subset_umap[(subset_umap["algorithm"] == algo) & (subset_umap["Normalised"] == norm_state)]
            ax.grid(alpha=0.3)

            if not subset.empty:
                sns.scatterplot(
                    data=subset,
                    x="silhouette_score",
                    y="AMI_score",
                    color=algo_colors.get(algo, "#999999"),
                    s=100, alpha=0.9, edgecolor="black", ax=ax
                )

                # Add Pearson correlation annotation
                row = summary_df[
                    (summary_df["algorithm"] == algo) &
                    (summary_df["Normalised"] == norm_state) &
                    (summary_df["UMAP"] == umap_state)
                ]
                if not row.empty:
                    r = row["r"].values[0]
                    p = row["p_value"].values[0]
                    sig = row["significance"].values[0]
                    text = f"r = {r:.2f}\np = {p:.3f}\n({sig})"
                    ax.text(
                        0.97, 0.05, text,
                        transform=ax.transAxes,
                        fontsize=9,
                        ha="right", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
                    )

            ax.set_title(f"{algo.upper()} | {norm_state}", fontsize=12)
            ax.set_xlabel("Silhouette Score")
            ax.set_ylabel("AMI Score")

    fig.suptitle(f"AMI vs Silhouette — {umap_state}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
