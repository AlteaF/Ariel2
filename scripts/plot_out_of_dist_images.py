import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONSTANTS ---
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
MOSAIC_WIDTH_COUNT = 50 # Number of images per row (column count)
TARGET_IMAGE_SIZE = 128 # Target size for each thumbnail in the mosaic

# --- HELPER FUNCTIONS (UNCHANGED) ---

def extract_class_from_path(path):
    """
    Extract trailing digits from the filename (digits immediately before the extension).
    """
    basename = os.path.basename(path)
    if isinstance(basename, (bytes, bytearray)):
        basename = basename.decode('utf-8')
        
    name, _ext = os.path.splitext(basename)
    m = re.search(r'(\d+)$', name)
    if not m:
        return None 
    return int(m.group(1))

def decode_path(p):
    """Safely decodes path from bytes to string if necessary."""
    return p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p)

# ----------------------------------------------------------------------
# --- MAIN FUNCTION (MODIFIED) ---
# ----------------------------------------------------------------------

def create_misclassified_mosaic(txt_file_path, 
                                image_base_dir, # <--- NEW ARGUMENT
                                output_mosaic_path='misclassified_mosaic.jpg', 
                                max_images_per_class=MOSAIC_WIDTH_COUNT, 
                                thumb_size=TARGET_IMAGE_SIZE):
    """
    Reads a list of image paths from a TXT file, groups them by class, 
    and creates an image mosaic where each row represents a class.
    
    Args:
        txt_file_path (str): Path to the text file containing relative image paths.
        image_base_dir (str): The root directory where all the images are actually located.
        output_mosaic_path (str): Path to save the resulting mosaic image.
        max_images_per_class (int): Maximum number of images to plot per class.
        thumb_size (int): The height and width of each thumbnail in pixels.
    """
    
    if not Path(txt_file_path).exists():
        print(f"Error: TXT file not found at {txt_file_path}")
        return
        
    # Ensure the base directory exists
    if not Path(image_base_dir).is_dir():
        print(f"Error: Image base directory not found at {image_base_dir}")
        return

    # 1. Read and Group Image Paths
    image_paths_by_class = {}
    valid_classes = set(LABEL_MAP.keys())
    
    with open(txt_file_path, 'r') as f:
        # The paths in the TXT file are often relative paths or just filenames.
        # We only keep the basename (filename) to join it reliably with the base dir.
        # If the path in the TXT file is already just the filename, os.path.basename doesn't change it.
        # If the path is /path/to/img_3.jpg, this isolates img_3.jpg
        all_basenames = [os.path.basename(line.strip()) for line in f if line.strip()]

    print(f"Read {len(all_basenames)} paths from {txt_file_path}.")

    for basename in all_basenames:
        cls_id = extract_class_from_path(basename)
        
        if cls_id is not None and cls_id in valid_classes:
            if cls_id not in image_paths_by_class:
                image_paths_by_class[cls_id] = []
            # We store the basename here; the full path is constructed when loading the image.
            image_paths_by_class[cls_id].append(basename)
        # else: Path is skipped if class cannot be determined or is unknown
            
    sorted_classes = sorted(image_paths_by_class.keys())
    print(f"Found misclassified images for {len(sorted_classes)} classes.")

    # 2. Prepare the Mosaic Canvas
    num_rows = len(sorted_classes)
    mosaic_width = max_images_per_class * thumb_size
    mosaic_height = num_rows * thumb_size
    
    fig_width = mosaic_width / 100 
    fig_height = mosaic_height / 100
    
    fig, ax = plt.subplots(figsize=(fig_width + 3, fig_height), dpi=100) 
    mosaic_data = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    # 3. Process Images and Fill Mosaic Array
    for row_idx, cls_id in enumerate(sorted_classes):
        basenames = image_paths_by_class[cls_id]
        
        # Select a random sample of basenames
        np.random.shuffle(basenames) 
        sample_basenames = basenames[:max_images_per_class]

        for col_idx, basename in enumerate(sample_basenames):
            # --- CRITICAL MODIFICATION ---
            # Construct the full path by joining the base directory with the filename
            full_path = os.path.join(image_base_dir, basename)
            
            try:
                # Open image using PIL
                img = Image.open(full_path).convert('RGB') # Use the full path here
                # Resize image to target thumbnail size
                img_thumb = img.resize((thumb_size, thumb_size))
                
                # Convert thumbnail to NumPy array
                thumb_array = np.array(img_thumb)
                
                # Calculate coordinates and place the thumbnail
                y_start = row_idx * thumb_size
                y_end = y_start + thumb_size
                x_start = col_idx * thumb_size
                x_end = x_start + thumb_size
                
                mosaic_data[y_start:y_end, x_start:x_end, :] = thumb_array
                
            except FileNotFoundError:
                print(f"Could not find image at expected path: {full_path}. Skipping.")
            except Exception as e:
                print(f"Error processing image {full_path}: {e}. Skipping.")

    # 4. Display and Save Mosaic with Labels
    ax.imshow(mosaic_data)
    ax.axis('off') 
    
    # Set up class labels on the left 
    for row_idx, cls_id in enumerate(sorted_classes):
        label = LABEL_MAP.get(cls_id, f"Class {cls_id}")
        y_pos = row_idx * thumb_size + thumb_size / 2
        ax.text(-20, 
                y_pos, 
                f"{cls_id}: {label}",
                transform=ax.transData,
                ha='right', 
                va='center', 
                fontsize=8,
                weight='bold',
                color='black')

    fig.subplots_adjust(left=0.1, right=1.0, top=0.95, bottom=0.05)
    ax.set_title("Misclassified Embeddings Mosaic (Closest NOT to Class Median)", pad=20)
    
    plt.savefig(output_mosaic_path, bbox_inches='tight', pad_inches=0.2)
    print(f"\n✅ Mosaic saved successfully to: {output_mosaic_path}")
    
    # plt.show()


# --- EXAMPLE USAGE ---

# 1. Define the input, base directory, and output paths
input_txt_file = '../analysis_of_embedding_distance/misclassified_paths_dino.txt' 
image_directory = '../data' 
output_image_file = 'mosaics/misclassified_image_mosaic_dino9x50.jpg'

# 2. Run the function (Uncomment this block to run)
create_misclassified_mosaic(
    txt_file_path=input_txt_file, 
    image_base_dir=image_directory,
    output_mosaic_path=output_image_file
)