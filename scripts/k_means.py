from sklearn.cluster import KMeans
import numpy as np
import argparse
from tqdm import tqdm

def run_kmeans(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Run Kmeans clustering on image embeddings.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--n_clusters', type=int, default=10)    
    parser.add_argument('--output_file', type=str, default='k_means_clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    args = parser.parse_args()

    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    print("🔍 Running K-Means clustering...")
    cluster_labels = run_kmeans(embeddings, args.n_clusters)

    np.savez(args.output_file, embeddings=embeddings, image_paths=image_paths, cluster_labels=cluster_labels)
    print(f"✅ Saved clustered data to {args.output_file}")


if __name__ == "__main__":
    main()