import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, pinv, cond, LinAlgError # Added for Mahalanobis dependencies
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
        return -1 
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
            # Skip classes with no examples
            continue 
        median_embeddings[cls] = np.median(embeddings[mask], axis=0)
    return median_embeddings

# --- NEW HELPER FUNCTIONS FOR MAHALANOBIS DISTANCE ---

def _calculate_per_class_inv_covs_diagonal(embeddings, classes, unique_classes, epsilon=1e-6):
    """
    Calculates the PER-CLASS DIAGONAL Inverse Covariance Matrices.
    This mimics the logic in your second code block for stability.
    """
    inv_cov_matrices = {}
    print("\n--- Calculating Per-Class DIAGONAL Inverse Covariance Matrices ---")
    
    for cls in unique_classes:
        mask = (classes == cls)
        class_embeddings = embeddings[mask]
        num_samples = class_embeddings.shape[0]
        
        if num_samples < 2:
            print(f"⚠️ Warning: Class {cls} has only {num_samples} sample(s). Cannot estimate variance. Skipping or using identity.")
            # Fallback: Use Identity Matrix (Euclidean distance essentially)
            inv_cov_matrices[cls] = np.eye(class_embeddings.shape[1])
            continue
            
        # Calculate full covariance
        cov_matrix = np.cov(class_embeddings, rowvar=False) 
        
        # KEY STEP: Extract only the variance (diagonal) terms
        class_variances = np.diag(cov_matrix).copy() 
        
        # Check for near-zero variance dimensions and set to epsilon (for stable division)
        zero_variance_mask = class_variances <= epsilon
        class_variances[zero_variance_mask] = epsilon 
        
        # Inverse of a Diagonal matrix is Diagonal matrix with 1/variance
        inv_variances = 1.0 / class_variances
        
        # Create the inverse diagonal matrix V_inv
        inv_cov_matrices[cls] = np.diag(inv_variances)
        
    return inv_cov_matrices

def _calculate_mahalanobis_dists_vectorized(embeddings, medians_matrix, inv_cov_matrices, unique_classes):
    """
    Calculates the Mahalanobis Distance for all embeddings to all class medians.
    N: number of samples, C: number of classes, D: embedding dimension
    """
    N, D = embeddings.shape
    C = medians_matrix.shape[0]
    dists = np.zeros((N, C))

    # The mahalanobis function from scipy is not easily fully vectorized for N x C distance 
    # when V_inv changes (per-class V_inv), so we loop over samples (N) and classes (C).
    # Since the original code did a fully vectorized Euclidean distance, this is the main 
    # computational change.

    for i in range(N):
        x = embeddings[i]
        for j, pred_cls in enumerate(unique_classes):
            mu = medians_matrix[j] # Median of class j
            V_inv = inv_cov_matrices[pred_cls] # Inverse Covariance of class j
            
            # mahalanobis(u, v, VI)
            dists[i, j] = mahalanobis(x, mu, V_inv)
            
    print(f"--- Mahalanobis distance calculated for {N} samples to {C} class medians. ---")
    return dists

# --- MODIFIED MAIN FUNCTION ---

def create_closest_to_median_npz(input_npz_path, 
                                 output_npz_path):
    """
    Reads .npz file, computes class medians, and saves a new .npz file 
    containing only the embeddings and paths that are closest to their 
    respective class median, using Mahalanobis Distance.
    """
    print(f"Loading data from: {input_npz_path}")
    
    # --- Load Data and Extract Classes (UNCHANGED) ---
    data = np.load(input_npz_path, allow_pickle=True)
    embeddings = np.array(data['embeddings'])
    image_paths = np.array(data['image_paths'])

    if image_paths.dtype.kind in ('S', 'a', 'O'):
        image_paths = np.array([p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths])

    try:
        classes = np.array([extract_class_from_path(p) for p in image_paths], dtype=int)
    except Exception as e:
        print(f"Error extracting class from path: {e}")
        return

    valid_mask = (classes > 0)
    if valid_mask.sum() < len(classes):
        print(f"Warning: Skipping {len(classes) - valid_mask.sum()} samples with invalid class ID.")
        embeddings = embeddings[valid_mask]
        image_paths = image_paths[valid_mask]
        classes = classes[valid_mask]

    unique_classes = np.unique(classes)
    unique_classes.sort()
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    true_class_indices = np.array([class_to_idx[cls] for cls in classes])

    # --- Calculate Medians, Inverse Covariances, and Distances (MODIFIED) ---
    
    # 1. Calculate Medians
    median_embeddings_dict = calculate_median_embeddings(embeddings, classes, unique_classes)
    medians_matrix = np.vstack([median_embeddings_dict[cls] for cls in unique_classes]) # shape (C, D)

    # 2. Calculate Per-Class DIAGONAL Inverse Covariance Matrices
    inv_cov_matrices = _calculate_per_class_inv_covs_diagonal(embeddings, classes, unique_classes)

    # 3. Calculate Mahalanobis Distances
    dists = _calculate_mahalanobis_dists_vectorized(embeddings, medians_matrix, inv_cov_matrices, unique_classes)
    
    # --- Filter for Closest to Own Median (UNCHANGED LOGIC) ---
    
    # 1. Find the index of the closest median for each embedding
    # dists shape is (N, C). np.argmin finds the column index (0 to C-1) of the minimum distance.
    pred_indices = np.argmin(dists, axis=1)

    # 2. Identify the samples that are closest to their *own* class median
    closest_to_own_median_mask = (pred_indices == true_class_indices)
    
    # --- Create and Save the Filtered Dataset (UNCHANGED) ---
    
    filtered_embeddings = embeddings[closest_to_own_median_mask]
    filtered_image_paths = image_paths[closest_to_own_median_mask]
    
    np.savez(output_npz_path, 
             embeddings=filtered_embeddings, 
             image_paths=filtered_image_paths)
    
    print("-" * 50)
    print(f"Original samples: {len(embeddings)}")
    print(f"Filtered samples (Closest to Own Median - Mahalanobis): {len(filtered_embeddings)}")
    print(f"Percentage kept: {len(filtered_embeddings) / len(embeddings) * 100:.2f}%")
    print(f"✅ Filtered data saved to: {output_npz_path}")
    print("-" * 50)


# ----------------------------------------------------------------------
# --- EXAMPLE USAGE (The original usage remains) ---
# ----------------------------------------------------------------------

# 1. Define input/output paths for the filtering step
input_file = '../embeddings_filtered/dino_embeddings_median_mahalanobis.npz'
# Changed output name to reflect Mahalanobis distance
output_file_filtered = '../embeddings_filtered/dino_emb_mahala.npz'

# 2. Run the new function to create the filtered NPZ file
create_closest_to_median_npz(
    input_npz_path=input_file,
    output_npz_path=output_file_filtered
)