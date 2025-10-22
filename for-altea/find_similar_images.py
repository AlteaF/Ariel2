# --- CONFIG ---
DIR_A = "../freshly-extracted_MOSAR-dataset_v1_01-29-2025-20250130T093657Z-001/MOSAR-dataset_v1_01-29-2025/train/images"
DIR_B = "../freshly-extracted_MOSAR-dataset_v1_01-29-2025-20250130T093657Z-001/MOSAR-dataset_v1_01-29-2025/valid/images"
OUTPUT_CSV = "similar_images.csv"
OUTPUT_PLOTS_DIR = "matched_pairs"

SSIM_THRESHOLD = 0.90       # 1.0 = identical
FILESIZE_TOLERANCE = 0.01   # 1% size difference allowed
IMG_SIZE = (256, 256)       # Resize for SSIM comparison

import os
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_and_preprocess(image_path, size=IMG_SIZE):
    """Load image, convert to grayscale, resize."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img


def compare_images(img1, img2):
    """Compute structural similarity index (SSIM)."""
    score, _ = ssim(img1, img2, full=True)
    return score


def plot_and_save(img_path1, img_path2, output_path, score):
    """Plot two images side by side and save."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    for ax, path in zip(axs, [img_path1, img_path2]):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        short_path = "/".join(path.split("/")[-3:])
        ax.set_title(short_path, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"SSIM Score: {score:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def process_one_image(args):
    """Worker: find best match in Dir_B for one image from Dir_A."""
    name_a, path_a, sizes_a, images_b, sizes_b = args
    img_a = load_and_preprocess(path_a)
    if img_a is None:
        return None

    best_score = -1
    best_name_b = None

    for name_b in images_b:
        size_a, size_b = sizes_a[name_a], sizes_b[name_b]
        if abs(size_a - size_b) / max(size_a, size_b) > FILESIZE_TOLERANCE:
            continue

        path_b = os.path.join(DIR_B, name_b)
        img_b = load_and_preprocess(path_b)
        if img_b is None:
            continue

        score = compare_images(img_a, img_b)
        if score > best_score:
            best_score = score
            best_name_b = name_b

    if best_score >= SSIM_THRESHOLD:
        out_path = os.path.join(
            OUTPUT_PLOTS_DIR,
            f"{os.path.splitext(name_a)[0]}_{best_name_b}.png"
        )
        plot_and_save(path_a, os.path.join(DIR_B, best_name_b), out_path, best_score)
        return (name_a, best_name_b, best_score)
    return None


def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # Gather image files
    images_a = [f for f in os.listdir(DIR_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_b = [f for f in os.listdir(DIR_B) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    sizes_a = {f: os.path.getsize(os.path.join(DIR_A, f)) for f in images_a}
    sizes_b = {f: os.path.getsize(os.path.join(DIR_B, f)) for f in images_b}

    print(f"🔍 Comparing {len(images_a)} images from Dir_A vs {len(images_b)} from Dir_B using {cpu_count()} cores...")

    tasks = [(name_a, os.path.join(DIR_A, name_a), sizes_a, images_b, sizes_b) for name_a in images_a]

    matches = []
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, tasks), total=len(tasks), desc="Processing"):
            if result:
                matches.append(result)

    # Save results to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_A", "Image_B", "SSIM_Score"])
        writer.writerows(matches)

    print(f"\n✅ Found {len(matches)} visually similar image pairs.")
    print(f"📁 CSV saved: {OUTPUT_CSV}")
    print(f"🖼️  Comparison plots: {OUTPUT_PLOTS_DIR}")


if __name__ == "__main__":
    main()
