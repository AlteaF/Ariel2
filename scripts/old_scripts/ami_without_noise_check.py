import os
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import argparse
import re

def extract_ground_truth_labels(image_paths):
    """Extract class IDs directly from cropped image filenames."""
    true_labels = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        tokens = re.findall(r'\d+', basename)
        if not tokens:
            print(f"⚠️ Warning: No numeric token found in filename: {filename}")
            true_labels.append(-1)
            continue
        class_id = int(tokens[-1])  # Last number is assumed to be the class ID
        true_labels.append(class_id)

    return true_labels


def compute_ami(cluster_labels, true_labels):
    paired = [(c, t) for c, t in zip(cluster_labels, true_labels) if t != -1]
    if not paired:
        raise ValueError("No valid labels found for AMI.")
    c_labels, t_labels = zip(*paired)
    return adjusted_mutual_info_score(t_labels, c_labels)

def main():
    parser = argparse.ArgumentParser(description="Compute AMI score from cluster and YOLO labels.")
    parser.add_argument('--clustered_file', type=str, required=True, help='Path to .npz with image paths + cluster labels')
    args = parser.parse_args()

    data = np.load(args.clustered_file, allow_pickle=True)
    image_paths = data['image_paths']
    cluster_labels = data['cluster_labels']

    true_labels = extract_ground_truth_labels(image_paths)
    ami_score = compute_ami(cluster_labels, true_labels)

    print(f"\n✅ Adjusted Mutual Information (AMI): {ami_score:.4f}")

if __name__ == "__main__":
    main()
