import pandas as pd
import glob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# --- Load and process experiments ---
folder = "../metrics_results_km"
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

    global_row = df[df["type"] == "global"]
    if not global_row.empty:
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

# --- Compute Pearson correlations per algorithm ---
results = []
for algo, group in df_all.groupby("algorithm"):
    if len(group) < 3:
        print(f"⚠️ Not enough data points for {algo}, skipping correlation.")
        continue

    pearson_corr, pearson_p = pearsonr(group["AMI_score"], group["silhouette_score"])
    pearson_sig = "Significant" if pearson_p < 0.05 else "Not significant"

    results.append({
        "algorithm": algo,
        "Pearson(AMI*Silhouette)": pearson_corr,
        "Pearson_p_value": pearson_p,
        "Pearson_significance": pearson_sig
    })

summary_df = pd.DataFrame(results)
summary_df.to_csv("ami_silhouette_pearson_correlation_per_algorithm.csv", index=False)
print(summary_df)

# --- Visualization ---
sns.set(style="whitegrid")

# Define color palettes for algorithms
algo_colors = {
    "k_means": ["#FFB6A0", "#FF6F61", "#D63C00", "#A62100"],          # reds
    "agglomerative": ["#B7E4C7", "#74C69D", "#40916C", "#1B4332"],   # greens
    "hdbscan": ["#A0C4FF", "#5390D9", "#004E98", "#012A4A"],          # blues
    "other": ["#C0C0C0"]                                             # grey fallback
}

# Assign color per point (depending on UMAP & Normalisation)
def assign_color(row):
    colors = algo_colors.get(row["algorithm"], ["#C0C0C0"])
    idx = 0
    if row["UMAP"] == "UMAP":
        idx += 2
    if row["Normalised"] == "Normalised":
        idx += 1
    idx = min(idx, len(colors) - 1)
    return colors[idx]

df_all["color"] = df_all.apply(assign_color, axis=1)

# --- Scatter plot ---
plt.figure(figsize=(10, 10))
for algo, group in df_all.groupby("algorithm"):
    plt.scatter(group["silhouette_score"], group["AMI_score"],
                color=group["color"], s=90, alpha=0.9, edgecolor="black",
                label=f"{algo}")

plt.xlabel("Global Silhouette Score")
plt.ylabel("Global AMI Score")
plt.title("AMI vs Silhouette Scores per Algorithm\nColor encodes UMAP & Normalisation")
plt.grid(alpha=0.3)

# --- Add correlation annotations per algorithm ---
for i, row in summary_df.iterrows():
    algo = row["algorithm"]
    subset = df_all[df_all["algorithm"] == algo]
    x_pos = subset["silhouette_score"].max() - 0.005
    y_pos = subset["AMI_score"].max() - 0.1

    text = (f"{algo}: r = {row['Pearson(AMI*Silhouette)']:.2f} "
            f"({row['Pearson_significance']})")
    plt.text(x_pos, y_pos, text, fontsize=9, alpha=0.8, verticalalignment='top')

# --- Add legend for hue coding ---
legend_elements = [
    Patch(facecolor="#FFB6A0", edgecolor="black", label="k-means | No UMAP, Not Normalised"),
    Patch(facecolor="#FF6F61", edgecolor="black", label="k-means | No UMAP, Normalised"),
    Patch(facecolor="#D63C00", edgecolor="black", label="k-means | UMAP, Not Normalised"),
    Patch(facecolor="#A62100", edgecolor="black", label="k-means | UMAP, Normalised"),

    Patch(facecolor="#B7E4C7", edgecolor="black", label="Agglomerative | No UMAP, Not Normalised"),
    Patch(facecolor="#74C69D", edgecolor="black", label="Agglomerative | No UMAP, Normalised"),
    Patch(facecolor="#40916C", edgecolor="black", label="Agglomerative | UMAP, Not Normalised"),
    Patch(facecolor="#1B4332", edgecolor="black", label="Agglomerative | UMAP, Normalised"),

    Patch(facecolor="#A0C4FF", edgecolor="black", label="HDBSCAN | No UMAP, Not Normalised"),
    Patch(facecolor="#5390D9", edgecolor="black", label="HDBSCAN | No UMAP, Normalised"),
    Patch(facecolor="#004E98", edgecolor="black", label="HDBSCAN | UMAP, Not Normalised"),
    Patch(facecolor="#012A4A", edgecolor="black", label="HDBSCAN | UMAP, Normalised"),
]

plt.legend(handles=legend_elements, title="Algorithm | UMAP & Normalisation",
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, title_fontsize=9)

plt.tight_layout()
plt.show()
