import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots
import umap
import argparse
import plotly.express as px

def main():
    parser = argparse.ArgumentParser(description="Visualize clustered embeddings in 2D and 3D.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to .npz file with reduced embeddings and cluster labels')
    parser.add_argument('--clusterer_name', type=str, required=True, help='name of the clusterer so that the title is correct')
    parser.add_argument('--output_prefix', type=str, default='cluster_vis', help='Prefix for output plots (without extension)')
    args = parser.parse_args()

    # Load data
    data = np.load(args.input_file, allow_pickle=True)
    embeddings = data['embeddings']
    cluster_labels = data['cluster_labels']

    # ---- 2D Visualization ----
    print("🔹 Reducing embeddings to 2D for visualization...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=cluster_labels, cmap='tab10', s=20, alpha=0.8
    )
    plt.title(f"UMAP + {args.clusterer_name} Clustering (2D projection)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_2d.png", dpi=300)
    plt.show()

    # ---- 3D Visualization ----
    print("🔹 Reducing embeddings to 3D for visualization...")
    reducer_3d = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer_3d.fit_transform(embeddings)

    # Matplotlib 3D static plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c=cluster_labels, cmap='tab10', s=20, alpha=0.8
    )
    ax.set_title(f"UMAP + {args.clusterer_name} Clustering (3D projection)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    fig.colorbar(p, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_3d.png", dpi=300)
    plt.show()

    # Optional: Interactive 3D Plot (Plotly)
    print("🔹 Generating interactive 3D Plotly visualization...")
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=cluster_labels.astype(str),
        title=f"UMAP + {args.clusterer_name} Clustering (Interactive 3D)",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.write_html(f"{args.output_prefix}_3d_interactive.html")
    fig.show()

    print(f"✅ Saved visualizations as {args.output_prefix}_2d.png, {args.output_prefix}_3d.png, and {args.output_prefix}_3d_interactive.html")

if __name__ == "__main__":
    main()
