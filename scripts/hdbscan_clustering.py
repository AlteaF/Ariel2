import numpy as np
import argparse
import hdbscan
from tqdm import tqdm

def run_hdbscan(embeddings, min_cluster_size=10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Run HDBSCAN clustering on image embeddings.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--min_cluster_size', type=int, default=10, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--output_file', type=str, default='clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    args = parser.parse_args()

    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    print("🔍 Running HDBSCAN clustering...")
    cluster_labels = run_hdbscan(embeddings, args.min_cluster_size)

    np.savez(args.output_file, embeddings=embeddings, image_paths=image_paths, cluster_labels=cluster_labels)
    print(f"✅ Saved clustered data to {args.output_file}")

if __name__ == "__main__":
    main()
