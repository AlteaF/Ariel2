import numpy as np
import os
import re
from collections import Counter
from sklearn.metrics import adjusted_mutual_info_score

def extract_ground_truth_labels(image_paths):
    """Integrated: Extract class IDs directly from cropped image filenames."""
    true_labels = []
    for image_path in image_paths:
        if isinstance(image_path, bytes): image_path = image_path.decode('utf-8')
        filename = os.path.basename(image_path)
        basename = os.path.splitext(filename)[0]
        # Using your re.findall logic: the last number is the class ID
        tokens = re.findall(r'\d+', basename)
        if not tokens:
            true_labels.append(-1)
            continue
        true_labels.append(int(tokens[-1]))
    return np.array(true_labels)

def analyze_clusters(npz_path, output_txt):
    # 1. Load Data
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    cluster_labels = data['cluster_labels']

    # 2. Extract Ground Truth and Calculate AMI
    true_labels = extract_ground_truth_labels(image_paths)
    
    # Filter valid pairs for AMI (excluding -1 noise/invalid)
    valid_mask = (cluster_labels != -1) & (true_labels != -1)
    if not np.any(valid_mask):
        ami_score = 0.0
    else:
        ami_score = adjusted_mutual_info_score(true_labels[valid_mask], cluster_labels[valid_mask])

    # 3. Write Report
    with open(output_txt, 'w') as f:
        f.write("FISH SPECIES CLUSTERING: COMPREHENSIVE ANALYSIS\n")
        f.write(f"Source: {npz_path}\n")
        f.write("="*85 + "\n")
        f.write(f"GLOBAL METRIC: Adjusted Mutual Information (AMI): {ami_score:.4f}\n")
        f.write(f"Total Samples: {len(image_paths)} | Valid for AMI: {np.sum(valid_mask)}\n")
        f.write("="*85 + "\n\n")

        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1: continue # Skip noise cluster if present
            
            mask = (cluster_labels == cluster_id)
            c_embeddings = embeddings[mask]
            c_labels = true_labels[mask]
            
            total_points = len(c_labels)
            overall_counts = Counter(c_labels)
            maj_class, maj_count = overall_counts.most_common(1)[0]
            overall_purity = (maj_count / total_points) * 100

            f.write(f"ANALYSIS FOR CLUSTER {cluster_id}\n")
            f.write(f"Total Points: {total_points} | Cluster Purity: {overall_purity:.2f}% (Majority: Class {maj_class})\n")
            f.write("-" * 65 + "\n")

            # Calculate Centroid and Euclidean Distances
            centroid = np.mean(c_embeddings, axis=0)
            distances = np.linalg.norm(c_embeddings - centroid, axis=1)

            # Sort and Decile
            sorted_indices = np.argsort(distances)
            sorted_gt = c_labels[sorted_indices]
            decile_size = total_points // 10

            for i in range(10):
                start = i * decile_size
                end = (i + 1) * decile_size if i < 9 else total_points
                
                chunk = sorted_gt[start:end]
                if len(chunk) == 0: continue
                
                chunk_counts = Counter(chunk)
                d_maj_class, d_maj_count = chunk_counts.most_common(1)[0]
                d_purity = (d_maj_count / len(chunk)) * 100

                counts_str = ", ".join([f"C{k}:{chunk_counts[k]}" for k in sorted(chunk_counts.keys())])
                header = f"Decile {i+1} ({i*10:2d}%-{(i+1)*10:2d}%):"
                f.write(f"{header:<22} Purity: {d_purity:6.2f}% | {counts_str}\n")
            
            f.write("\n" + "="*85 + "\n\n")

    print(f"✅ Analysis Complete. AMI: {ami_score:.4f}. Report saved to: {output_txt}")
# Run it
analyze_clusters('../filtered_clusters/agglomerative_clusters_dino_rgb.npz', '../cluster_decile_breakdown/dino_rgb_ac_euclidean.txt')