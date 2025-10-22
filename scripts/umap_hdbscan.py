import hdbscan
import umap
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run HDBSCAN clustering on image embeddings after UMAP.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--min_cluster_size', type=int, default=10, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--output_file', type=str, default='clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    args = parser.parse_args()

    # Load embeddings
    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    # Reduce dimensions with UMAP
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=50,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
        n_jobs=1  # prevent numba parallel crash
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Save results
    np.savez(args.output_file, embeddings=reduced_embeddings, image_paths=image_paths, cluster_labels=cluster_labels)
    print(f"✅ Saved clustered data to {args.output_file}")


if __name__ == "__main__":
    main()
