import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONSTANTS AND HELPER FUNCTIONS (UNCHANGED) ---

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

def extract_class_from_path(path):
    """
    Extract trailing digits from the filename (digits immediately before the extension).
    """
    basename = os.path.basename(path)
    name, _ext = os.path.splitext(basename)
    # find trailing digits at end of name
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError(f"No trailing digit(s) found in filename '{basename}'")
    return int(m.group(1))

def calculate_median_embeddings(embeddings, classes, unique_classes):
    """
    embeddings: (N, D) array
    classes: (N,) array of ints
    unique_classes: sorted list/array of classes to compute medians for
    returns dict {cls: median_vector}
    """
    median_embeddings = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            raise ValueError(f"No examples found for class {cls}")
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
    return median_embeddings


def analyze_embeddings_and_save_csv(npz_path, 
                                    out_csv_path='median_classification_summary_dino.csv',
                                    misclassified_txt_path='misclassified_paths.txt', # <--- NEW ARGUMENT
                                    tie_atol=1e-8, 
                                    distribute_ties_fractionally=False):
    """
    Reads .npz with keys 'embeddings' and 'image_paths'.
    Produces CSV summarizing median classification and saves paths of misclassified
    embeddings to a TXT file.
    """
    
    # --- Load Data ---
    data = np.load(npz_path, allow_pickle=True)
    if 'embeddings' not in data or 'image_paths' not in data:
        raise KeyError("NPZ must contain 'embeddings' and 'image_paths' arrays")

    embeddings = np.array(data['embeddings'])
    image_paths = np.array(data['image_paths'])

    # If image_paths are bytes, decode to str
    if image_paths.dtype.kind in ('S', 'a', 'O'):
        image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths])

    # Extract classes
    classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)

    # Unique classes sorted
    unique_classes = np.unique(classes)
    unique_classes.sort()
    
    # Create a mapping from class ID to its index in unique_classes array (for fast lookup)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    true_class_indices = np.array([class_to_idx[cls] for cls in classes]) # (N,) array of indices

    # --- Calculate Medians and Distances ---
    median_embeddings = calculate_median_embeddings(embeddings, classes, unique_classes)
    medians_matrix = np.vstack([median_embeddings[cls] for cls in unique_classes])  # shape (C, D)

    # Vectorized distance calculation: dists shape (N, C)
    diff = embeddings[:, None, :] - medians_matrix[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # --- Classification and Misclassified Path Collection ---
    
    # Find the index of the closest median for each embedding
    # np.argmin finds the index of the minimum distance along the C axis (axis=1)
    # The result, pred_indices, has shape (N,)
    pred_indices = np.argmin(dists, axis=1)

    # Use vectorized comparison to find samples where the predicted index is NOT the true index
    # We ignore ties here for simplicity, counting any sample not closest to its own median as misclassified
    # The original loop below handles ties for the CSV summary, but for the TXT file, 
    # we'll save any sample whose minimum distance is not its own class.
    
    # Create a mask for misclassified samples
    # A sample is correctly classified if the index of its closest median is the index of its true class
    misclassified_mask = (pred_indices != true_class_indices)
    
    # Collect the paths of the misclassified samples
    misclassified_paths = image_paths[misclassified_mask]

    # Save misclassified paths to TXT file
    with open(misclassified_txt_path, 'w') as f:
        # Write one path per line
        f.write('\n'.join(misclassified_paths))
        
    print(f"Total misclassified samples (not closest to own class median): {len(misclassified_paths)}")
    print(f"Saved misclassified paths to: {misclassified_txt_path}")
    
    # --- Build CSV Summary (Original Logic, adjusted to use indices) ---
    
    pred_cols = [str(c) for c in unique_classes]
    rows = {str(c): np.zeros(len(unique_classes), dtype=float) for c in unique_classes}
    ties_counts = {str(c): 0 for c in unique_classes}
    totals = {str(c): 0 for c in unique_classes}
    
    # Note: The original loop logic is kept here to correctly handle fractional ties
    # if distribute_ties_fractionally is True, which the vectorized approach cannot do easily.
    for i, true_cls in enumerate(classes):
        true_cls_str = str(true_cls)
        totals[true_cls_str] += 1
        distances = dists[i]  # shape (C,)
        min_dist = distances.min()
        
        # find indices within tolerance -> ties
        tie_mask = np.isclose(distances, min_dist, atol=tie_atol)
        tie_indices = np.nonzero(tie_mask)[0]
        
        if tie_indices.size > 1:
            # tie
            ties_counts[true_cls_str] += 1
            if distribute_ties_fractionally:
                frac = 1.0 / tie_indices.size
                for idx in tie_indices:
                    pred_cls_str = str(unique_classes[idx])
                    rows[true_cls_str][idx] += frac
            # else: ties are not counted in 'pred_X' columns
        else:
            pred_idx = int(tie_indices[0])
            rows[true_cls_str][pred_idx] += 1.0

    # Build DataFrame
    df_rows = []
    for cls in unique_classes:
        cls_str = str(cls)
        row = { 'true_class': cls,
                'total_instances': int(totals[cls_str]),
                'ties': int(ties_counts[cls_str]) }
        # counts for each predicted class
        for j, pred_cls in enumerate(unique_classes):
            row[f'pred_{pred_cls}'] = float(rows[cls_str][j])
        # correct count and pct
        correct = row[f'pred_{cls}']
        row['correct'] = float(correct)
        row['correct_pct'] = float(correct) / row['total_instances'] if row['total_instances'] > 0 else 0.0
        df_rows.append(row)

    cols_order = ['true_class', 'total_instances', 'ties'] + [f'pred_{c}' for c in unique_classes] + ['correct', 'correct_pct']
    df = pd.DataFrame(df_rows)[cols_order].sort_values('true_class').reset_index(drop=True)

    # Save CSV
    df.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV to: {out_csv_path}")
    return df


# --- PLOTTING FUNCTION (UNCHANGED) ---

def plot_pretty_confusion_matrices(df):
    """
    df: DataFrame returned from analyze_embeddings_and_save_csv()
    Creates 2 seaborn heatmaps:
      - raw counts
      - normalized per true class
    """
    # ... (Plotting code remains the same) ...
    # Extract true classes (rows)
    true_classes = df["true_class"].values

    # Extract predicted class columns
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_classes = [int(c.replace("pred_", "")) for c in pred_cols]

    # Raw confusion matrix
    cm_raw = df[pred_cols].values.astype(float)

    # Normalized confusion matrix (row-wise)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)

    # --------------------------
    # Raw counts heatmap
    # --------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_raw,
                annot=True,
                fmt=".0f",
                cmap="Reds",
                xticklabels=[LABEL_MAP[c] for c in pred_classes],
                yticklabels=[LABEL_MAP[c] for c in true_classes])

    plt.title("Heat map of DINOv2 embeddings \n median analysis", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    # Note: Saving path is hardcoded; you might want to adjust
    # plt.savefig("../analysis_of_embedding_distance/heatmap_dino.pdf") 
    plt.show()

    # --------------------------
    # Normalized heatmap
    # --------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                xticklabels=[LABEL_MAP[c] for c in pred_classes],
                yticklabels=[LABEL_MAP[c] for c in true_classes])

    plt.title("Heat map (row normalised) of DINOv2 embeddings \n median analysis", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    # Note: Saving path is hardcoded; you might want to adjust
    # plt.savefig("../analysis_of_embedding_distance/normalised_heatmap_dino.pdf")
    plt.show()


#Example usage (assuming the NPZ file exists):
df = analyze_embeddings_and_save_csv('../embeddings_files/dino_embeddings.npz',
                                    out_csv_path='../analysis_of_embedding_distance/median_classification_summary_dino.csv',
                                    misclassified_txt_path='../analysis_of_embedding_distance/misclassified_paths_dino.txt',
                                    tie_atol=1e-8,
                                    distribute_ties_fractionally=False)
print(df)
plot_pretty_confusion_matrices(df)