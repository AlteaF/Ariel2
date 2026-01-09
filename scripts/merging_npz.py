import numpy as np
from pathlib import Path

def merge_npz_files(file_paths, output_filename='merged_data.npz'):
    """
    Merges the 'embeddings' and 'image_paths' arrays from multiple .npz files
    into a single new .npz file.

    Args:
        file_paths (list): A list of strings or Path objects for the .npz files to merge.
        output_filename (str): The name of the resulting merged .npz file.
    """
    # Initialize lists to store the arrays from all files
    all_embeddings = []
    all_image_paths = []

    # Iterate through all input files
    for file_path in file_paths:
        try:
            # Load the data from the .npz file
            data = np.load(file_path, allow_pickle=True)

            # Check if the expected keys exist
            if 'embeddings' not in data:
                print(f"Warning: '{file_path}' is missing the 'embeddings' array. Skipping.")
                continue
            if 'image_paths' not in data:
                print(f"Warning: '{file_path}' is missing the 'image_paths' array. Skipping.")
                continue
            
            # Append the arrays to our lists
            all_embeddings.append(data['embeddings'])
            all_image_paths.append(data['image_paths'])
            
            # Close the file handle
            data.close()

        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")
            
    # --- Concatenate the Arrays ---
    
    # Check if we successfully loaded any data
    if not all_embeddings:
        print("No valid data loaded. Aborting merge.")
        return

    # Use np.concatenate to join the arrays along the first axis (rows/entries)
    try:
        merged_embeddings = np.concatenate(all_embeddings, axis=0)
        # For object arrays (like strings/paths), np.concatenate is usually fine
        merged_image_paths = np.concatenate(all_image_paths, axis=0)
    except ValueError as e:
        print(f"Error during concatenation. Check array shapes: {e}")
        return

    # --- Save the Merged Data ---
    
    # Save the combined arrays into a new compressed .npz file
    np.savez_compressed(
        output_filename, 
        embeddings=merged_embeddings, 
        image_paths=merged_image_paths
    )
    
    print(f"\n✅ Successfully merged {len(file_paths)} files into '{output_filename}'.")
    print(f"Total merged embeddings shape: {merged_embeddings.shape}")
    print(f"Total merged image paths shape: {merged_image_paths.shape}")

files_to_merge = ["../embeddings_files/embeddings_files_split_data/dino_embeddings_train_gray.npz", "../embeddings_files/embeddings_files_split_data/dino_embeddings_test_gray.npz"]
output_file = "../embeddings_files/dino_embeddings_gray.npz"    
merge_npz_files(files_to_merge, output_file)