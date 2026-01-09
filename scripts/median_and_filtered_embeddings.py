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
    CORRECTED: Extract the last single digit immediately before the file extension.
    Example: 'dir/foo_123_7.jpg' -> 7
    Example: 'dir/another-345.png' -> 5
    """
    filename = os.path.basename(path)
    # Regex to find a single digit (\d) that is followed by a dot and the extension [^\.]*$.
    # This is more robust than searching for trailing digits in the base name alone.
    match = re.search(r"(\d)(?=\.[^\.]*$)", filename, flags=re.IGNORECASE)
    
    if not match:
        # Return -1 if no single digit is found before the extension
        return -1 
    return int(match.group(1))

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
            # Skip classes with no examples
            continue 
        # Use np.median for robust centering
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
    return median_embeddings


def create_closest_to_median_npz(input_npz_path, 
                                 output_npz_path):
    """
    Reads .npz file, computes class medians, and saves a new .npz file 
    containing only the embeddings and paths that are closest to their 
    respective class median.
    
    Args:
        input_npz_path (str): Path to the original .npz file.
        output_npz_path (str): Path to save the new filtered .npz file.
    """
    print(f"Loading data from: {input_npz_path}")
    
    # --- Load Data ---
    data = np.load(input_npz_path, allow_pickle=True)
    if 'embeddings' not in data or 'image_paths' not in data:
        raise KeyError("NPZ must contain 'embeddings' and 'image_paths' arrays")

    embeddings = np.array(data['embeddings'])
    image_paths = np.array(data['image_paths'])

    # If image_paths are bytes, decode to str
    if image_paths.dtype.kind in ('S', 'a', 'O'):
        image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths])

    # Extract classes (uses the corrected function)
    try:
        classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)
    except Exception as e:
        print(f"Error extracting class from path: {e}")
        return

    # Filter out samples where class extraction failed (if any)
    valid_mask = (classes > 0)
    if valid_mask.sum() < len(classes):
        print(f"Warning: Skipping {len(classes) - valid_mask.sum()} samples with invalid class ID (<= 0).")
        embeddings = embeddings[valid_mask]
        image_paths = image_paths[valid_mask]
        classes = classes[valid_mask]

    # Unique classes sorted
    unique_classes = np.unique(classes)
    unique_classes.sort()
    
    # Create a mapping from class ID to its index in unique_classes array (for fast lookup)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    # (N,) array of indices corresponding to the row in medians_matrix for the true class
    true_class_indices = np.array([class_to_idx[cls] for cls in classes])

    # --- Calculate Medians and Distances ---
    median_embeddings_dict = calculate_median_embeddings(embeddings, classes, unique_classes)
    
    # Create the matrix of medians for vectorized distance calculation
    # Medians are ordered by unique_classes
    medians_matrix = np.vstack([median_embeddings_dict[cls] for cls in unique_classes])  # shape (C, D)

    # Vectorized distance calculation: Euclidean Distance
    # dists shape (N, C) -> dists[i, j] is the distance of sample i to the median of class j
    diff = embeddings[:, None, :] - medians_matrix[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # --- Filter for Closest to Own Median ---
    
    # 1. Find the index of the closest median for each embedding
    # pred_indices is the index (0 to C-1) of the median that is CLASSIFIED as closest.
    pred_indices = np.argmin(dists, axis=1)

    # 2. Identify the samples that are closest to their *own* class median
    # We keep the sample if the closest median (pred_indices) is the median of its true class (true_class_indices).
    closest_to_own_median_mask = (pred_indices == true_class_indices)
    
    # --- Create the Filtered Dataset ---
    
    filtered_embeddings = embeddings[closest_to_own_median_mask]
    filtered_image_paths = image_paths[closest_to_own_median_mask]
    
    # --- Save the New NPZ File ---
    os.makedirs(os.path.dirname(output_npz_path) or '.', exist_ok=True)
    np.savez(output_npz_path, 
             embeddings=filtered_embeddings, 
             image_paths=filtered_image_paths)
    
    print("-" * 50)
    print(f"Original samples: {len(embeddings)}")
    print(f"Filtered samples (Closest to Own Median): {len(filtered_embeddings)}")
    if len(embeddings) > 0:
        print(f"Percentage kept: {len(filtered_embeddings) / len(embeddings) * 100:.2f}%")
    print(f"✅ Filtered data saved to: {output_npz_path}")
    print("-" * 50)


# ----------------------------------------------------------------------
# --- EXAMPLE USAGE ---
# ----------------------------------------------------------------------

# 1. Define input/output paths for the filtering step
input_file = '../normalised_embeddings/dino_normalised_embeddings.npz'
output_file_filtered = '../embeddings_filtered/dino_embeddings_norm_median.npz'

# 2. Run the new function to create the filtered NPZ file
create_closest_to_median_npz(
    input_npz_path=input_file,
    output_npz_path=output_file_filtered
)