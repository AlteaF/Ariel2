# --- CONFIG ---
IMAGE_DIR = "../dataset/data"  # single directory
OUTPUT_CSV = "similar_images.csv"
OUTPUT_PLOTS_DIR = "matched_pairs"

PHASH_THRESHOLD = 8         # smaller = stricter; 0 means nearly identical
SSIM_THRESHOLD = 0.90       # 1.0 = identical
FILESIZE_TOLERANCE = 0.02   # 2% size difference allowed
IMG_SIZE = (256, 256)       # Resize for SSIM comparison

import os
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# --- IMAGE UTILS ---
def load_and_preprocess(image_path, size=IMG_SIZE):
    """Load image, convert to grayscale, resize."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img


def compute_phash(image_path):
    """Compute perceptual hash (pHash) for an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    try:
        phash = cv2.img_hash.pHash(img)
        return phash
    except Exception:
        return None


def phash_distance(hash1, hash2):
    """Compute Hamming distance between two perceptual hashes."""
    return np.count_nonzero(hash1 != hash2)


def compare_images(img1, img2):
    """Compute structural similarity index (SSIM)."""
    score, _ = ssim(img1, img2, full=True)
    return score


def plot_and_save(img_path1, img_path2, output_path, score):
    """Plot two images side by side and save."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, path in zip(axs, [img_path1, img_path2]):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        short_path = os.path.basename(path)
        ax.set_title(short_path, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"SSIM Score: {score:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


# --- WORKER FUNCTION ---
def compare_pair(args):
    """Compare a single image pair with pHash prefilter."""
    (name_a, path_a, name_b, path_b, sizes, phashes) = args

    size_a, size_b = sizes[name_a], sizes[name_b]
    if abs(size_a - size_b) / max(size_a, size_b) > FILESIZE_TOLERANCE:
        return None

    hash_a, hash_b = phashes[name_a], phashes[name_b]
    if hash_a is None or hash_b is None:
        return None

    # Quick check: skip very different images
    dist = phash_distance(hash_a, hash_b)
    if dist > PHASH_THRESHOLD:
        return None

    # Compute expensive SSIM check
    img_a = load_and_preprocess(path_a)
    img_b = load_and_preprocess(path_b)
    if img_a is None or img_b is None:
        return None

    score = compare_images(img_a, img_b)
    if score >= SSIM_THRESHOLD:
        out_path = os.path.join(
            OUTPUT_PLOTS_DIR,
            f"{os.path.splitext(name_a)[0]}_{os.path.splitext(name_b)[0]}.png"
        )
        plot_and_save(path_a, path_b, out_path, score)
        return (name_a, name_b, dist, score)
    return None


# --- MAIN ---
def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # Gather images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sizes = {f: os.path.getsize(os.path.join(IMAGE_DIR, f)) for f in image_files}

    print(f"🔍 Found {len(image_files)} images. Computing perceptual hashes...")

    # Precompute perceptual hashes
    phashes = {}
    for f in tqdm(image_files, desc="Computing pHashes"):
        phashes[f] = compute_phash(os.path.join(IMAGE_DIR, f))

    # Prepare all unique pairs
    tasks = []
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            name_a, name_b = image_files[i], image_files[j]
            path_a = os.path.join(IMAGE_DIR, name_a)
            path_b = os.path.join(IMAGE_DIR, name_b)
            tasks.append((name_a, path_a, name_b, path_b, sizes, phashes))

    print(f"⚙️  Comparing {len(tasks):,} possible pairs using {cpu_count()} cores...")

    matches = []
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for result in tqdm(pool.imap_unordered(compare_pair, tasks), total=len(tasks), desc="Filtering & Matching"):
            if result:
                matches.append(result)

    # Save results
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_1", "Image_2", "pHash_Distance", "SSIM_Score"])
        writer.writerows(matches)

    print(f"\n✅ Found {len(matches)} visually similar image pairs.")
    print(f"📁 CSV saved: {OUTPUT_CSV}")
    print(f"🖼️  Visual matches saved in: {OUTPUT_PLOTS_DIR}")


if __name__ == "__main__":
    main()
