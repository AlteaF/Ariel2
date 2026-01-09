import numpy as np
import matplotlib.pyplot as plt
import argparse
from pacmap import PaCMAP
from collections import defaultdict, Counter
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
import os
import re
import numpy.random as npr


# -------------------------------------------------------
# ✔️ Hard-coded label map (edit this to your class names)
# -------------------------------------------------------
LABEL_MAP = {
        1: "Starfish",
        2: "Crab",
        3: "Black goby",
        4: "Wrasse",
        5: "Two-spotted goby",
        6: "Cod",
        7: "Painted goby",
        8: "Sand eel",
        9: "Whiting"
    }
# -----------------------------------

IMAGE_BASE_DIR = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train"


def load_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"]
    return embeddings, image_paths


def extract_numeric_label_from_path(path):
    """
    Extract last sequence of digits before file extension.
    Supports .jpg/.jpeg/.png/.JPG etc.
    """
    filename = os.path.basename(path)
    match = re.search(r"(\d+)(?=\.(?:jpe?g|png)$)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract label from filename: {filename}")
    return int(match.group(1))


def resolve_image_path(img_path, npz_base_dir):
    """
    Always try IMAGE_BASE_DIR first.
    """
    # 1) Try IMAGE_BASE_DIR + basename
    candidate = os.path.join(IMAGE_BASE_DIR, os.path.basename(img_path))
    if os.path.exists(candidate):
        return candidate

    # 2) Try path as-is
    if os.path.exists(img_path):
        return img_path

    # 3) Try relative to npz directory
    candidate2 = os.path.join(npz_base_dir, img_path)
    if os.path.exists(candidate2):
        return candidate2

    # 4) Try basename inside npz directory
    candidate3 = os.path.join(npz_base_dir, os.path.basename(img_path))
    if os.path.exists(candidate3):
        return candidate3

    return None


def plot_pacmap_with_images(
    embeddings, image_paths, labels,
    output_path=None,
    samples_per_label=1,
    n_neighbors=10,
    n_components=2,
    random_state=42,
    show_images=True,
    npz_base_dir="."
):
    print(f"Running PacMAP on {len(embeddings)} vectors...")

    reducer = PaCMAP(
    n_components=n_components,
    n_neighbors=n_neighbors,
    random_state=random_state)
    pacmap_results = reducer.fit_transform(embeddings, init="pca")

    print("Embedding complete. Plotting...")

    counts = Counter(labels)
    print("Counts per label:")
    for lab, c in counts.items():
        print(f"  {lab}: {c}")

    unique_labels = sorted(counts.keys())
    cmap = plt.get_cmap("tab20")
    label_to_color = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique_labels)}

    plt.figure(figsize=(12, 8))

    # Plot each label separately
    for lab in unique_labels:
        idxs = [i for i, x in enumerate(labels) if x == lab]
        coords = pacmap_results[idxs]
        plt.scatter(coords[:, 0], coords[:, 1],
                    color=label_to_color[lab], s=20, alpha=0.7,
                    label=f"{lab} ({len(idxs)})")

    # Legend with word labels
    legend_handles = [Patch(color=label_to_color[lab], label=f"{lab} ({counts[lab]})")
                      for lab in unique_labels]
    plt.legend(handles=legend_handles, title="Labels",
               bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add thumbnails
    if show_images:
        print("Adding thumbnails...")
        grouped = defaultdict(list)
        for i, lab in enumerate(labels):
            grouped[lab].append(i)

        rng = npr.default_rng(random_state)

        added = 0
        missing = 0

        for lab, idxs in grouped.items():
            sample = min(samples_per_label, len(idxs))
            chosen = rng.choice(idxs, size=sample, replace=False)

            for idx in chosen:
                resolved = resolve_image_path(image_paths[idx], npz_base_dir)
                if resolved is None:
                    missing += 1
                    print(f"  Missing image: {image_paths[idx]}")
                    continue

                try:
                    img = Image.open(resolved).convert("RGB")
                    img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                    imagebox = OffsetImage(np.asarray(img), zoom=1)
                    ab = AnnotationBbox(
                        imagebox,
                        (pacmap_results[idx, 0], pacmap_results[idx, 1]),
                        frameon=False
                    )
                    plt.gca().add_artist(ab)
                    added += 1
                except Exception as e:
                    missing += 1
                    print(f"  Error loading thumbnail for {resolved}: {e}")

        print(f"Thumbnails added: {added}, missing: {missing}")

    plt.title("PacMAP Visualization")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="PacMAP visualization with filename-based labels")
    parser.add_argument("--embedding_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--samples_per_label", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--show_images", action="store_true")

    args = parser.parse_args()

    embeddings, image_paths = load_npz_data(args.embedding_file)
    npz_base_dir = os.path.dirname(os.path.abspath(args.embedding_file))

    # Extract numeric labels → map to word labels
    numeric_labels = []
    for p in image_paths:
        try:
            numeric_labels.append(extract_numeric_label_from_path(p))
        except ValueError:
            numeric_labels.append(None)

    labels = [LABEL_MAP.get(n, "unknown") for n in numeric_labels]

    plot_pacmap_with_images(
        embeddings=embeddings,
        image_paths=image_paths,
        labels=labels,
        output_path=args.output_file,
        samples_per_label=args.samples_per_label,
        n_neighbors=args.neighbors,
        show_images=args.show_images,
        npz_base_dir=npz_base_dir
    )


if __name__ == "__main__":
    main()
