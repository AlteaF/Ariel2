import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# --- CONSTANTS ---
LABEL_MAP = {
    1: "Starfish", 2: "Crab", 3: "Black goby", 4: "Wrasse", 
    5: "Two-spotted goby", 6: "Cod", 7: "Painted goby", 8: "Sand eel", 
    9: "Whiting"
}
# Output directory for plots and the new TXT files
OUTPUT_DIR = "../euclidean_median/" # Changed output directory name for consistency
FILTERED_EMB_DIR = "../euclidean_filtered_embeddings/" # Changed output directory name for consistency
# EPSILON is no longer strictly needed for Euclidean, but kept for general safety
EPSILON = 1e-6 


# --- HELPER FUNCTIONS ---

def extract_class_from_path(path):
    """
    Extract the single digit label (1-9) immediately preceded by an underscore 
    and followed by the file extension.
    Example: 'dir/foo_123_7.jpg' -> 7
    """
    filename = os.path.basename(path)
    # Regex: find a single digit (\d) that is preceded by '_' and followed 
    # by a dot and the extension (non-dot characters till end $).
    match = re.search(r"\_(\d)(?=\.[^\.]*$)", filename, flags=re.IGNORECASE)
    
    if not match:
        return -1 
    return int(match.group(1))

def calculate_median(embeddings, classes, unique_classes):
    """
    Calculates the median vector for each unique class, used as the centroid.
    
    Returns:
        median_embeddings (dict): {cls: median_vector}
    """
    median_embeddings = {}
    for cls in unique_classes:
        mask = (classes == cls)
        if mask.sum() == 0:
            print(f"Warning: No examples found for class {cls}. Skipping calculation.")
            continue 
        
        # Use the median as the centroid for robustness against outliers
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
        
    return median_embeddings

def calculate_euclidean_distance_sq(x_data, median_vector):
    """
    Calculates the squared Euclidean Distance: 
    sum((x_i - mu_i)^2)
    """
    # Squared difference: (x - mu)^2
    diff_sq = (x_data - median_vector)**2
    
    # Sum over the dimensions (D)
    return np.sum(diff_sq, axis=-1)


def create_confusion_df(embeddings, classes, medians_dict, unique_classes, tie_atol=1e-8):
    """
    Classifies embeddings based on the Squared Euclidean Distance to the provided medians 
    and returns a DataFrame suitable for confusion matrix plotting.
    """
    N = len(embeddings)
    C = len(unique_classes)
    
    dists_sq = np.zeros((N, C))
    
    for j, cls in enumerate(unique_classes):
        # *** MODIFIED: Using Euclidean Distance function ***
        dists_sq[:, j] = calculate_euclidean_distance_sq(
            embeddings, 
            medians_dict[cls]
        )
    
    min_dists = dists_sq.min(axis=1, keepdims=True)
    tie_mask = np.isclose(dists_sq, min_dists, atol=tie_atol)
    
    counts = {str(r): np.zeros(C, dtype=float) for r in unique_classes}
    
    for i in range(N):
        true_cls = classes[i]
        if true_cls not in unique_classes:
            continue
            
        tie_indices = np.nonzero(tie_mask[i])[0]
        
        # Tie handling (distribute ties fractionally)
        frac = 1.0 / tie_indices.size
        for pred_idx in tie_indices:
            counts[str(true_cls)][pred_idx] += frac

    df_rows = []
    unique_classes_str = [str(c) for c in unique_classes]
    for cls in unique_classes:
        row = {'true_class': cls}
        for j, pred_cls_str in enumerate(unique_classes_str):
            row[f'pred_{pred_cls_str}'] = counts[str(cls)][j]
        df_rows.append(row)

    cols_order = ['true_class'] + [f'pred_{c}' for c in unique_classes]
    df = pd.DataFrame(df_rows)[cols_order].sort_values('true_class').reset_index(drop=True)
    return df

def plot_confusion_matrix(df, title_suffix, filename_suffix, distance_type="Euclidean"): # *** MODIFIED: Distance type label ***
    """Plots a normalized confusion matrix from a classification DataFrame."""
    true_classes = df["true_class"].values
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_classes = [int(c.replace("pred_", "")) for c in pred_cols]
    
    cm_raw = df[pred_cols].values.astype(float)
    cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                xticklabels=[LABEL_MAP.get(c, str(c)) for c in pred_classes],
                yticklabels=[LABEL_MAP.get(c, str(c)) for c in true_classes])

    # *** MODIFIED: Title reflects Euclidean Distance ***
    plt.title(f"{distance_type} Classification Accuracy (Row Normalized)\n{title_suffix}", fontsize=16) 
    plt.xlabel("Predicted Class (Closest Median)", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{filename_suffix}.pdf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot to: {output_path}")

# --- MAIN FUNCTION ---

def run_median_analysis_and_filter(input_npz_path):
    """
    Performs the 4 steps: initial analysis, plotting, filtering by ED, and 
    final plotting.
    """
    print(f"1. Loading and preparing data from: {input_npz_path}")
    
    # Load Data
    data = np.load(input_npz_path, allow_pickle=True)
    embeddings = np.array(data['embeddings'])
    image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in data['image_paths']])
    
    # Extract and filter classes
    classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)
    valid_mask = (classes > 0)
    
    # Filter out samples where class extraction failed
    if valid_mask.sum() < len(classes):
        print(f"   ⚠️ Warning: Skipping {len(classes) - valid_mask.sum()} samples with invalid class ID.")
        embeddings = embeddings[valid_mask]
        image_paths = image_paths[valid_mask]
        classes = classes[valid_mask]
        
    unique_classes = np.unique(classes)
    unique_classes.sort()

    # 2a. Calculate Original Medians (the stable parameters)
    # *** MODIFIED: Only calculating medians, not variance ***
    median_dict = calculate_median(embeddings, classes, unique_classes)
    
    print(f"   Calculated {len(unique_classes)} class medians for Euclidean Distance.")

    # 2b. Run Initial Median Classification & Plot Confusion Matrix
    # *** MODIFIED: Calling create_confusion_df without variance_dict ***
    print("2. Running initial Euclidean classification on all embeddings...")
    df_original = create_confusion_df(embeddings, classes, median_dict, unique_classes)
    plot_confusion_matrix(df_original, 
                          "Before Filtering (Euclidean Distance)", 
                          "euclidean_analysis_original_all") # *** MODIFIED: Filename suffix ***

    # 3a. Filter Embeddings (Identify samples closest to their OWN median by ED)
    print("3. Filtering embeddings: Keeping only those closest to their own class median...")

    # The true class index in the unique_classes list
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    
    # Initialize distances matrix (N, C)
    N = len(embeddings)
    C = len(unique_classes)
    dists_sq = np.zeros((N, C))
    
    # Calculate squared ED for all samples against all medians
    for j, cls in enumerate(unique_classes):
        # *** MODIFIED: Using Euclidean Distance function ***
        dists_sq[:, j] = calculate_euclidean_distance_sq(
            embeddings, 
            median_dict[cls]
        )
    
    # Find the index of the closest median (by Euclidean Distance)
    pred_indices = np.argmin(dists_sq, axis=1)

    # Filtering mask: Keep if the closest median index equals the true class index
    true_class_indices = np.array([class_to_idx[cls] for cls in classes])
    closest_to_own_median_mask = (pred_indices == true_class_indices)
    
    filtered_embeddings = embeddings[closest_to_own_median_mask]
    filtered_image_paths = image_paths[closest_to_own_median_mask]

    # --- Save paths to TXT files (Kept same logic) ---
    
    # Paths of the data that was KEPT
    kept_paths = image_paths[closest_to_own_median_mask]
    kept_file_path = os.path.join(OUTPUT_DIR, "kept_image_paths.txt")
    
    # Paths of the data that was FILTERED OUT (the opposite of the mask)
    filtered_out_paths = image_paths[~closest_to_own_median_mask]
    filtered_out_file_path = os.path.join(OUTPUT_DIR, "filtered_out_image_paths.txt")
    
    # Save the kept paths
    with open(kept_file_path, 'w') as f:
        f.write('\n'.join(kept_paths))
    print(f"✅ Saved KEPT paths to: **{kept_file_path}**")
    
    # Save the filtered out paths
    with open(filtered_out_file_path, 'w') as f:
        f.write('\n'.join(filtered_out_paths))
    print(f"✅ Saved FILTERED OUT paths to: **{filtered_out_file_path}**")


    # 3b. Save Filtered NPZ file (Original logic retained)
    output_filename = os.path.basename(input_npz_path).replace('.npz', '_filtered_euclidean.npz') # *** MODIFIED: Filename suffix ***
    output_npz_path = os.path.join(FILTERED_EMB_DIR, output_filename)
    os.makedirs(FILTERED_EMB_DIR, exist_ok=True)
    
    np.savez(output_npz_path, 
             embeddings=filtered_embeddings, 
             image_paths=filtered_image_paths)
    
    print("-" * 50)
    print(f"Original samples: {len(embeddings)}")
    print(f"Filtered samples (Closest to Own Median by ED): {len(filtered_embeddings)}") # *** MODIFIED: Distance type ***
    if len(embeddings) > 0:
        print(f"Percentage kept: {len(filtered_embeddings) / len(embeddings) * 100:.2f}%")
    print(f"✅ Filtered embeddings saved to: **{output_npz_path}**")
    print("-" * 50)

    # 4. Plot the remaining (filtered) data using the ORIGINAL medians
    # *** MODIFIED: Calling create_confusion_df without variance_dict ***
    print("4. Plotting final confusion matrix using the ORIGINAL Euclidean parameters...")
    filtered_classes = classes[closest_to_own_median_mask]
    
    df_filtered = create_confusion_df(filtered_embeddings, filtered_classes, median_dict, unique_classes)
    plot_confusion_matrix(df_filtered, 
                          "After Filtering (Classified with ORIGINAL Euclidean Parameters)", # *** MODIFIED: Title reflects Euclidean Distance ***
                          "euclidean_analysis_filtered_100_percent") # *** MODIFIED: Filename suffix ***


# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Median Analysis and Filtering using Euclidean Distance.") # *** MODIFIED: Distance type ***
    parser.add_argument("--embedding_file", 
                        type=str, 
                        default='../embeddings_files/dino_embeddings.npz', 
                        help="Path to the input .npz file with embeddings and image_paths.")
    args = parser.parse_args()

    # Ensure all output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FILTERED_EMB_DIR, exist_ok=True)
    
    try:
        run_median_analysis_and_filter(args.embedding_file)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found at {args.embedding_file}")
        print("Please check the path or provide a correct path via the --embedding_file argument.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")