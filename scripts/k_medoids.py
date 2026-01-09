from sklearn_extra.cluster import KMedoids # <-- New import
import numpy as np
import argparse
from tqdm import tqdm
import os # <-- Added os import for output file path handling

def run_kmedoids(embeddings, n_clusters=10, random_state=42):
    """
    Runs K-Medoids clustering (PAM) on the embeddings.
    K-Medoids uses actual data points (medoids) as cluster centers.
    """
    # KMedoids defaults to Euclidean distance, which is suitable for embeddings.
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state, metric='euclidean', max_iter=300)
    
    print(f"   Fitting K-Medoids with {n_clusters} clusters...")
    kmedoids.fit(embeddings)
    
    # Use the fitted model to predict the labels
    cluster_labels = kmedoids.labels_
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Run K-Medoids clustering on image embeddings.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-Medoids.')    
    parser.add_argument('--output_file', type=str, default='k_medoids_clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')
    args = parser.parse_args()

    # --- Load Data ---
    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    print(f"🔍 Running K-Medoids clustering on {len(embeddings)} samples...")
    
    # --- Run K-Medoids ---
    cluster_labels = run_kmedoids(
        embeddings, 
        n_clusters=args.n_clusters, 
        random_state=args.random_state
    )

    # --- Save Results ---
    
    # Ensure output directory exists before saving
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez(
        args.output_file, 
        embeddings=embeddings, 
        image_paths=image_paths, 
        cluster_labels=cluster_labels
    )
    print(f"✅ Saved clustered data (embeddings, paths, labels) to {args.output_file}")


if __name__ == "__main__":
    
    main()