import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load your CSV
df = pd.read_csv("../cluster_scores/ami_silhouette_values_per_experiment.csv")

# --- Extract metadata from experiment names ---
def extract_metadata(name):
    name = name.lower()
    algo = None
    if "k_means" in name or "km" in name:
        algo = "k_means"
    elif "hdbscan" in name:
        algo = "hdbscan"
    elif "agglomerative" in name or name.startswith("ac_") or "_ac_" in name:
        algo = "agglomerative"
    else:
        algo = "other"

    umap = "UMAP" if "umap" in name else "No UMAP"
    norm = "Normalised" if "normalised" in name else "Not normalised"

    return pd.Series([algo, umap, norm])

df[["algorithm", "umap", "normalised"]] = df["experiment"].apply(extract_metadata)

# --- Color mapping ---
palette = {
    "k_means": {
        "No UMAP|Not normalised": "#FFB6A0",
        "No UMAP|Normalised": "#FF6F61",
        "UMAP|Not normalised": "#D63C00",
        "UMAP|Normalised": "#A62100"
    },
    "agglomerative": {
        "No UMAP|Not normalised": "#B7E4C7",
        "No UMAP|Normalised": "#74C69D",
        "UMAP|Not normalised": "#40916C",
        "UMAP|Normalised": "#1B4332"
    },
    "hdbscan": {
        "No UMAP|Not normalised": "#A0C4FF",
        "No UMAP|Normalised": "#5390D9",
        "UMAP|Not normalised": "#004E98",
        "UMAP|Normalised": "#012A4A"
    }
}

# --- Plot ---
plt.figure(figsize=(10, 10))

for algo in df["algorithm"].unique():
    subset = df[df["algorithm"] == algo]
    for (u, n), group in subset.groupby(["umap", "normalised"]):
        color = palette.get(algo, {}).get(f"{u}|{n}", "gray")
        plt.scatter(group["silhouette_score"], group["AMI_score"],
                    color=color, s=90, alpha=0.9, edgecolor="black",
                    label=f"{algo} - {u} - {n}")

plt.title("AMI vs Silhouette Scores per Experiment\nColored by Algorithm, UMAP, and Normalisation")
plt.xlabel("Silhouette Score")
plt.ylabel("AMI Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
