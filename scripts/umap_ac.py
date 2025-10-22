from sklearn.cluster import AgglomerativeClustering
import umap 
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np 
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Agglomerative clustering on image embeddings after UMAP.")
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for Agglomerative Clustering')
    parser.add_argument('--output_file', type=str, default='clustered_embeddings.npz', help='Output .npz file to store cluster labels')
    args = parser.parse_args()

    data = np.load(args.embedding_file, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']


    # test different number of neighbours and different numbers of components. 
    # I'd start with 50 components
    reducer = umap.UMAP(n_neighbors=30, n_components=50, min_dist=0.0, metric="euclidean",random_state=42) # n_components is the number of dimensions to decrease to. n_neighbors how many we want to look at to clutser. 
    reduced_embeddings = reducer.fit_transform(embeddings)


    clusterer = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="average")
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    np.savez(args.output_file, embeddings=reduced_embeddings, image_paths=image_paths, cluster_labels=cluster_labels)
    print(f"✅ Saved clustered data to {args.output_file}")

if __name__ == "__main__":
    main()


