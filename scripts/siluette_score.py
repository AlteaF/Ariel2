import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples


def compute_silhouette(embeddings, cluster_labels):
    """Compute average and per-sample Silhouette scores."""
    valid_mask = cluster_labels != -1  # Remove HDBSCAN noise points
    if np.sum(valid_mask) < 2 or len(set(cluster_labels[valid_mask])) < 2:
        raise ValueError("Not enough valid clusters for Silhouette score.")

    avg_score = silhouette_score(embeddings[valid_mask], cluster_labels[valid_mask])
    sample_scores = silhouette_samples(embeddings[valid_mask], cluster_labels[valid_mask])

    return avg_score, valid_mask, sample_scores


def save_scores_csv(image_paths, cluster_labels, silhouette_scores, csv_file):
    """Save per-sample silhouette scores into a CSV file."""
    df = pd.DataFrame({
        "image_path": image_paths,
        "cluster_label": cluster_labels,
        "silhouette_score": silhouette_scores
    })
    df.to_csv(csv_file, index=False)
    print(f"💾 Saved per-sample silhouette scores to {csv_file}")


def plot_silhouette(embeddings, cluster_labels, silhouette_scores, out_file=None):
    """2D scatterplot of embeddings colored by silhouette score."""
    # If embeddings > 2D, project down with PCA for plotting
    from sklearn.decomposition import PCA
    if embeddings.shape[1] > 2:
        reduced = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        reduced = embeddings

    valid_mask = cluster_labels != -1
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        reduced[valid_mask, 0], reduced[valid_mask, 1],
        c=silhouette_scores, cmap="viridis", s=10, alpha=0.7
    )
    plt.colorbar(sc, label="Silhouette Score")
    plt.title("Silhouette Scores per Sample")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"📊 Saved silhouette scatterplot to {out_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute Silhouette score from clustered embeddings.")
    parser.add_argument('--clustered_file', type=str, required=True, help='Path to .npz with embeddings + cluster labels')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to save per-sample silhouette scores as CSV')
    parser.add_argument('--plot_file', type=str, default=None, help='Optional path to save scatterplot as PNG')
    args = parser.parse_args()

    # Load clustered data
    data = np.load(args.clustered_file, allow_pickle=True)
    embeddings = data['embeddings']
    cluster_labels = data['cluster_labels']
    image_paths = data.get('image_paths', np.arange(len(cluster_labels)))

    # Compute silhouette
    avg_score, valid_mask, sample_scores = compute_silhouette(embeddings, cluster_labels)
    print(f"\n✅ Average Silhouette Score: {avg_score:.4f}")
    print(f"ℹ️  Per-sample scores computed for {np.sum(valid_mask)} non-noise points")

    # Fill silhouette_scores (NaN for noise points)
    silhouette_scores = np.full_like(cluster_labels, np.nan, dtype=np.float64)
    silhouette_scores[valid_mask] = sample_scores

    # Save per-sample scores into CSV
    save_scores_csv(image_paths, cluster_labels, silhouette_scores, args.csv_file)

    # Optional scatterplot
    if args.plot_file:
        plot_silhouette(embeddings, cluster_labels, silhouette_scores, args.plot_file)


if __name__ == "__main__":
    main()
