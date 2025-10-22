import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def run_agglomerative(embeddings, n_clusters=10):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Run Agglomerative clustering on image embeddings.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for Agglomerative Clustering')
    parser.add_argument('--output_file', type=str, default='clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    args = parser.parse_args()

    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    print("🔍 Running Agglomerative clustering...")
    cluster_labels = run_agglomerative(embeddings, args.n_clusters)

   

    np.savez(args.output_file, embeddings=embeddings, image_paths=image_paths, cluster_labels=cluster_labels)
    print(f"✅ Saved clustered data to {args.output_file}")

if __name__ == "__main__":
    main()
