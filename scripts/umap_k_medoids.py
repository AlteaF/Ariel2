import umap
from sklearn_extra.cluster import KMedoids
import numpy as np
import argparse
import os

def run_kmedoids(embeddings, n_clusters=9, random_state=42):
    """
    Runs K-Medoids clustering (PAM) on the embeddings.
    """
    # KMedoids defaults to Euclidean distance, which is suitable for embeddings.
    kmedoids = KMedoids(
        n_clusters=n_clusters, 
        random_state=random_state, 
        metric='euclidean', 
        max_iter=300
    )
    
    print(f"   Fitting K-Medoids with {n_clusters} clusters...")
    kmedoids.fit(embeddings)
    
    # Use the fitted model to predict the labels
    cluster_labels = kmedoids.labels_
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Run K-Medoids clustering on image embeddings after UMAP reduction.")
    
    # --- Input/Output Arguments ---
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--output_file', type=str, default='umap_kmedoids_clustered.npz', help='Output .npz file to store reduced embeddings and cluster labels')
    
    # --- UMAP Arguments ---
    parser.add_argument('--n_components', type=int, default=50, help='Number of dimensions for UMAP reduction.')
    parser.add_argument('--n_neighbors', type=int, default=30, help='Number of nearest neighbors for UMAP.')
    
    # --- K-Medoids Arguments ---
    parser.add_argument('--n_clusters', type=int, default=9, help='Number of clusters for K-Medoids.')    
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility across UMAP and K-Medoids.')
    
    args = parser.parse_args()

    # --- Load Data ---
    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # --- Reduce dimensions with UMAP ---
    print(f"📉 Reducing dimensions with UMAP to {args.n_components} components...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=0.0,
        metric="euclidean",
        random_state=args.random_state,
        n_jobs=1  # Prevent numba parallel crash
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    
    # --- Run K-Medoids on Reduced Embeddings ---
    print(f"🔍 Running K-Medoids clustering on {len(reduced_embeddings)} reduced samples...")
    cluster_labels = run_kmedoids(
        reduced_embeddings, 
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
        embeddings=reduced_embeddings,  # Storing the reduced embeddings
        image_paths=image_paths, 
        cluster_labels=cluster_labels
    )
    print(f"✅ Saved clustered data (reduced embeddings, paths, labels) to {args.output_file}")


if __name__ == "__main__":
    
    main()