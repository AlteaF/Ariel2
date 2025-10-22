#!/usr/bin/env python3
"""
Visualize clustered images by selecting representative samples
from the center to the boundary of each cluster and creating
10x10 mosaics.

Usage:
    python visualize_clusters.py \
        --clustered_file path/to/clusters.npz \
        --output_dir results/ \
        --prefix experiment1
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_cluster_data(npz_path):
    """Load embeddings, image paths, and cluster labels from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    cluster_labels = data['cluster_labels']
    return embeddings, image_paths, cluster_labels


def compute_cluster_centroid(embeddings):
    """Compute centroid of given embeddings."""
    return np.mean(embeddings, axis=0)


def select_representative_images(embeddings, image_paths, n_select=100):
    """
    Select n_select images evenly spaced by distance percentile
    from cluster center (closest to farthest).
    """
    centroid = compute_cluster_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    sorted_idx = np.argsort(distances)

    if len(sorted_idx) < n_select:
        # If fewer than 100 images, repeat as needed
        selected_idx = np.resize(sorted_idx, n_select)
    else:
        percentiles = np.linspace(0, len(sorted_idx) - 1, n_select).astype(int)
        selected_idx = sorted_idx[percentiles]

    selected_paths = [image_paths[i] for i in selected_idx]
    return selected_paths, selected_idx


def create_mosaic(image_paths, output_path, grid_size=(10, 10), image_size=(128, 128)):
    """
    Create and save a mosaic of given image paths using matplotlib.
    The central image corresponds to the cluster centroid (index 50).
    """
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, ax in enumerate(axes.flat):
        if i < len(image_paths):
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                img = img.resize(image_size)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
                print(f"⚠️ Could not open image {image_paths[i]}: {e}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize clustered images in mosaics.")
    parser.add_argument('--clustered_file', type=str, required=True,
                        help='Path to .npz file with embeddings, image_paths, and cluster_labels')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save mosaics and CSV summary')
    parser.add_argument('--prefix', type=str, default="clusters",
                        help='Prefix for output files (default: "clusters")')
    parser.add_argument('--n_select', type=int, default=100,
                        help='Number of images to select per cluster (default: 100)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size (pixels) of each image in mosaic grid (default: 128)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📂 Loading clustered data...")
    embeddings, image_paths, cluster_labels = load_cluster_data(args.clustered_file)

    df_records = []

    unique_clusters = sorted(np.unique(cluster_labels))
    print(f"🔍 Found {len(unique_clusters)} clusters")

    for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_image_paths = image_paths[cluster_mask]

        if len(cluster_embeddings) == 0:
            print(f"⚠️ Cluster {cluster_id} is empty. Skipping.")
            continue

        selected_paths, selected_idx = select_representative_images(
            cluster_embeddings, cluster_image_paths, args.n_select
        )

        mosaic_filename = f"{args.prefix}_cluster_{cluster_id}_mosaic.jpg"
        mosaic_path = os.path.join(args.output_dir, mosaic_filename)

        create_mosaic(selected_paths, mosaic_path, image_size=(args.image_size, args.image_size))

        for order_idx, img_path in enumerate(selected_paths):
            df_records.append({
                "cluster_id": cluster_id,
                "image_path": img_path,
                "order_index": order_idx,
                "mosaic_path": mosaic_path
            })

    summary_df = pd.DataFrame(df_records)
    csv_filename = f"{args.prefix}_selected_images_summary.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    summary_df.to_csv(csv_path, index=False)

    print(f"\n✅ Saved mosaics and summary CSV to: {args.output_dir}")
    print(f"📄 Summary file: {csv_path}")


if __name__ == '__main__':
    main()
