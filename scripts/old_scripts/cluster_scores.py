import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

def compute_metrics(embeddings, cluster_labels):
    """Compute global and per-cluster clustering metrics."""

    # Filter out noise (e.g., HDBSCAN assigns -1 to noise)
    mask = cluster_labels != -1
    X = embeddings[mask]
    labels = cluster_labels[mask]

    if len(set(labels)) <= 1:
        raise ValueError("Need at least 2 clusters (excluding noise) to compute metrics.")

    # --- Global scores ---
    sil_score = silhouette_score(X, labels)
    ch_index = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)

    # --- Per-cluster silhouette averages ---
    sil_samples = silhouette_samples(X, labels)
    cluster_silhouette = {
        cluster_id: sil_samples[labels == cluster_id].mean()
        for cluster_id in np.unique(labels)
    }

    results = {
        "global": {
            "silhouette_score": sil_score,
            "calinski_harabasz_index": ch_index,
            "davies_bouldin_index": db_index,
        },
        "per_cluster": cluster_silhouette,
    }
    return results


def save_results(results, output_csv):
    """Save global and per-cluster metrics into a CSV file."""
    rows = []

    # Global row
    rows.append({
        "type": "global",
        "cluster_id": "all",
        "silhouette_score": results["global"]["silhouette_score"],
        "calinski_harabasz_index": results["global"]["calinski_harabasz_index"],
        "davies_bouldin_index": results["global"]["davies_bouldin_index"],
    })

    # Per-cluster rows
    for cluster_id, score in results["per_cluster"].items():
        rows.append({
            "type": "cluster",
            "cluster_id": cluster_id,
            "silhouette_score": score,
            "calinski_harabasz_index": "",
            "davies_bouldin_index": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved clustering evaluation results to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Compute clustering evaluation metrics (Silhouette, CH, DBI).")
    parser.add_argument("--clustered_file", type=str, required=True, help="Path to .npz file with embeddings + cluster labels")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to output CSV file with results")
    args = parser.parse_args()

    # Load clustered embeddings
    data = np.load(args.clustered_file, allow_pickle=True)
    embeddings = data["embeddings"]
    cluster_labels = data["cluster_labels"]

    # Compute metrics
    results = compute_metrics(embeddings, cluster_labels)

    # Save to CSV
    save_results(results, args.csv_file)


if __name__ == "__main__":
    main()
