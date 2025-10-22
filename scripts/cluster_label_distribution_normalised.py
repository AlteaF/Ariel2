import argparse
import numpy as np
import os
import re
import pandas as pd
from collections import defaultdict
total_images_per_class_9 = { 1:18540, 2:5241, 3:3802, 4:972, 5:700, 6:613, 7:549,8:257, 9:203}
# Dictionary of total images per class (can be swapped out as needed)
#total_images_per_class_4 = { 11: 11669,1: 18543, 2: 5213, 10: 4934 }

def extract_ground_truth_labels(image_paths):
    true_labels = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        tokens = re.findall(r'\d+', basename)
        if not tokens:
            print(f"⚠️ Warning: No numeric token found in filename: {filename}")
            true_labels.append(-1)
            continue
        class_id = int(tokens[-1])
        true_labels.append(class_id)
    return true_labels

def analyze_clusters(cluster_labels, true_labels):
    cluster_summary = defaultdict(lambda: defaultdict(int))
    for cluster, label in zip(cluster_labels, true_labels):
        if label != -1:
            cluster_summary[cluster][label] += 1
    return cluster_summary

def print_cluster_summary(cluster_summary):
    print("\n📊 Cluster-Class Distribution (count and % of total images per class):")
    for cluster_id in sorted(cluster_summary):
        print(f"\nCluster {cluster_id}:")
        for class_id, count in sorted(cluster_summary[cluster_id].items()):
            total_class_images = total_images_per_class_9.get(class_id)
            if total_class_images is None:
                print(f"  ⚠️ Skipping Class {class_id} (not in total_images dict)")
                continue
            percent = (count / total_class_images) * 100 if total_class_images > 0 else 0
            print(f"  Class {class_id}: {count} images ({percent:.2f}%)")

def save_summary_to_csv(cluster_summary, output_path):
    rows = []
    for cluster_id in sorted(cluster_summary):
        for class_id in sorted(cluster_summary[cluster_id]):
            count = cluster_summary[cluster_id][class_id]
            total_class_images = total_images_per_class_9.get(class_id)
            if total_class_images is None:
                continue  # skip if not in the total images dict
            percent = (count / total_class_images) * 100 if total_class_images > 0 else 0
            rows.append({
                'Cluster': cluster_id,
                'Class': class_id,
                'Count': count,
                'Percentage of class (%)': round(percent, 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Summary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze class distribution within clusters.")
    parser.add_argument('--clustered_file', type=str, required=True, help='Path to .npz file with embeddings, image paths, and cluster labels')
    parser.add_argument('--save_csv', type=str, help='Path to save the cluster summary as a CSV')
    args = parser.parse_args()

    print("📥 Loading clustered data...")
    data = np.load(args.clustered_file, allow_pickle=True)
    image_paths = data['image_paths']
    cluster_labels = data['cluster_labels']

    print("🔍 Extracting ground truth labels from image paths...")
    true_labels = extract_ground_truth_labels(image_paths)

    print("📊 Analyzing clusters...")
    cluster_summary = analyze_clusters(cluster_labels, true_labels)

    print_cluster_summary(cluster_summary)

    if args.save_csv:
        save_summary_to_csv(cluster_summary, args.save_csv)

if __name__ == '__main__':
    main()
