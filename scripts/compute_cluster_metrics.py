import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_mutual_info_score


# ===============================================================
# Helper functions
# ===============================================================

def extract_ground_truth_labels(image_paths):
    """
    Extract class IDs directly from cropped image filenames.
    The last number in the filename is assumed to be the class ID.
    Example: 'fish_3.jpg' -> class 3
    """
    true_labels = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        tokens = re.findall(r'\d+', basename)
        if not tokens:
            print(f"⚠️ Warning: No numeric token found in filename: {filename}")
            true_labels.append(-1)
            continue
        class_id = int(tokens[-1])
        true_labels.append(class_id)

    return np.array(true_labels)


def compute_cluster_purity(cluster_labels, true_labels):
    """
    Compute per-cluster purity.
    Purity = (# of samples of the majority true class in the cluster) / (total samples in cluster)
    """
    purity = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # skip noise
            continue
        mask = cluster_labels == cluster_id
        true_sub = true_labels[mask]
        if len(true_sub) == 0 or np.all(true_sub == -1):
            purity[cluster_id] = np.nan
        else:
            counts = np.bincount(true_sub[true_sub != -1])
            purity[cluster_id] = counts.max() / counts.sum()
    return purity


def compute_metrics(embeddings, cluster_labels, image_paths):
    """
    Compute clustering metrics:
      - Global: Silhouette, AMI
      - Per-cluster: Silhouette, Purity
    """
    # Filter out noise points
    
    mask = cluster_labels != -1
    X = embeddings[mask]
    labels = cluster_labels[mask]
    paths = np.array(image_paths)[mask]

    if len(set(labels)) <= 1:
        raise ValueError("Need at least 2 clusters (excluding noise) to compute metrics.")

    # Extract ground-truth labels from filenames
    true_labels = extract_ground_truth_labels(paths)

    # --- Global metrics ---
    sil_global = silhouette_score(X, labels)
    ami_global = adjusted_mutual_info_score(true_labels, labels)

    # --- Per-cluster silhouette ---
    sil_samples = silhouette_samples(X, labels)
    sil_per_cluster = {
        cid: sil_samples[labels == cid].mean()
        for cid in np.unique(labels)
    }

    # --- Per-cluster purity ---
    purity_per_cluster = compute_cluster_purity(labels, true_labels)

    # --- Results ---
    results = {
        "global": {
            "silhouette_score": sil_global,
            "ami_score": ami_global,
        },
        "per_cluster": {
            cid: {
                "silhouette_score": sil_per_cluster[cid],
                "purity": purity_per_cluster.get(cid, np.nan),
                "n_samples": np.sum(labels == cid),
            }
            for cid in np.unique(labels)
        },
    }
    return results


def save_results(results, output_csv):
    """Save metrics to CSV (global + per cluster)."""
    rows = []

    # Global row
    rows.append({
        "type": "global",
        "cluster_id": "all",
        "silhouette_score": results["global"]["silhouette_score"],
        "ami_score": results["global"]["ami_score"],
        "purity": "",
        "n_samples": "",
    })

    # Per-cluster rows
    for cluster_id, scores in results["per_cluster"].items():
        rows.append({
            "type": "cluster",
            "cluster_id": cluster_id,
            "silhouette_score": scores["silhouette_score"],
            "ami_score": "",
            "purity": scores["purity"],
            "n_samples": scores["n_samples"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved clustering evaluation results to: {output_csv}")


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Silhouette, AMI, and Cluster Purity metrics.")
    parser.add_argument("--clustered_file", required=True, help="Path to .npz file with embeddings, cluster_labels, and image_paths")
    parser.add_argument("--csv_file", required=True, help="Output CSV file path")
    args = parser.parse_args()

    # Load data
    data = np.load(args.clustered_file, allow_pickle=True)
    embeddings = data["embeddings"]
    cluster_labels = data["cluster_labels"]
    image_paths = data["image_paths"]

    # Compute metrics
    results = compute_metrics(embeddings, cluster_labels, image_paths)

    # Print summary
    print("\n=== GLOBAL METRICS ===")
    print(f"Silhouette Score: {results['global']['silhouette_score']:.4f}")
    print(f"AMI Score:        {results['global']['ami_score']:.4f}")

    # Save to CSV
    save_results(results, args.csv_file)
