import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots
import umap
import argparse
import plotly.express as px

def main():
    parser = argparse.ArgumentParser(description="Visualize image embeddings using UMAP in 2D and 3D (no clustering).")
    parser.add_argument('--input_file', type=str, required=True, help='Path to .npz file with embeddings and image paths')
    parser.add_argument('--output_prefix', type=str, default='umap_vis', help='Prefix for output plots (without extension)')
    args = parser.parse_args()

    # Load data
    data = np.load(args.input_file, allow_pickle=True)
    embeddings = data['embeddings']

    print(f"✅ Loaded embeddings from {args.input_file} with shape {embeddings.shape}")

    # ---- 2D Visualization ----
    print("🔹 Reducing embeddings to 2D with UMAP...")
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c='gray', s=15, alpha=0.7
    )
    plt.title("UMAP Projection (2D)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_2d.png", dpi=300)
    plt.show()

    # ---- 3D Visualization ----
    print("🔹 Reducing embeddings to 3D with UMAP...")
    reducer_3d = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer_3d.fit_transform(embeddings)

    # Matplotlib 3D static plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
        c='gray', s=15, alpha=0.7
    )
    ax.set_title("UMAP Projection (3D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_3d.png", dpi=300)
    plt.show()

    # Optional: Interactive 3D Plot (Plotly)
    print("🔹 Generating interactive 3D Plotly visualization...")
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        title="UMAP Projection (Interactive 3D)",
        color_discrete_sequence=["gray"]
    )
    fig.write_html(f"{args.output_prefix}_3d_interactive.html")
    fig.show()

    print(f"✅ Saved visualizations as {args.output_prefix}_2d.png, {args.output_prefix}_3d.png, and {args.output_prefix}_3d_interactive.html")

if __name__ == "__main__":
    main()

