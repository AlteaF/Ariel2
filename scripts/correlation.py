import pandas as pd
import glob
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Folder containing your experiment CSV files
folder = "../cluster_scores"
files = glob.glob(f"{folder}/*.csv")

# Collect global AMI and silhouette from all experiments
data = []

for file in files:
    df = pd.read_csv(file)
    
    # Normalize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Skip files that don’t have the required columns
    if not {"type", "silhouette_score", "ami_score"}.issubset(df.columns):
        print(f"⚠️ Skipping {file}: missing required columns.")
        continue

    global_row = df[df["type"] == "global"]
    if not global_row.empty:
        ami = pd.to_numeric(global_row["ami_score"].values[0], errors="coerce")
        sil = pd.to_numeric(global_row["silhouette_score"].values[0], errors="coerce")
        if pd.notna(ami) and pd.notna(sil):
            data.append({
                "experiment": file.split("/")[-1].replace(".csv", ""),
                "AMI_score": ami,
                "silhouette_score": sil
            })

df_all = pd.DataFrame(data)

# Ensure enough data points
if len(df_all) < 3:
    raise ValueError("Need at least 3 experiments with valid global scores to compute correlation.")

# Compute correlations and p-values
pearson_corr, pearson_p = pearsonr(df_all["AMI_score"], df_all["silhouette_score"])
spearman_corr, spearman_p = spearmanr(df_all["AMI_score"], df_all["silhouette_score"])

# Statistical significance flags
pearson_sig = "Significant" if pearson_p < 0.05 else "Not significant"
spearman_sig = "Significant" if spearman_p < 0.05 else "Not significant"

# Save results
summary = pd.DataFrame({
    "Pearson(AMI*Silhouette)": [pearson_corr],
    "Pearson_p_value": [pearson_p],
    "Pearson_significance": [pearson_sig],
    "Spearman(AMI*Silhouette)": [spearman_corr],
    "Spearman_p_value": [spearman_p],
    "Spearman_significance": [spearman_sig]
})
summary.to_csv("ami_silhouette_correlation_across_experiments.csv", index=False)

# Save raw values
df_all.to_csv("ami_silhouette_values_per_experiment.csv", index=False)

# Print summary to terminal
print(summary)

# ---- Visualization ----
plt.figure(figsize=(10, 10))
plt.scatter(df_all["silhouette_score"], df_all["AMI_score"], s=70, alpha=0.7)
plt.xlabel("Global Silhouette Score")
plt.ylabel("Global AMI Score")

title = (
    f"AMI vs Silhouette across experiments\n"
    f"Pearson r={pearson_corr:.3f} (p={pearson_p:.3f}, {pearson_sig}), "
    f"Spearman ρ={spearman_corr:.3f} (p={spearman_p:.3f}, {spearman_sig})"
)
plt.title(title, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
