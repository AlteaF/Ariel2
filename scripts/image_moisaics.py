import os
import random
from PIL import Image

def extract_class_from_filename(filename):
    """Extract the class label from the filename (last digit before .jpg)."""
    base = os.path.splitext(filename)[0]
    return int(base[-1])  # Assumes class is the last digit

def load_images_from_folder(folder, target_size=(128, 128)):
    """Load images from folder and group by class."""
    class_images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            class_label = extract_class_from_filename(filename)
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(target_size)
            if class_label not in class_images:
                class_images[class_label] = []
            class_images[class_label].append(img)
    return class_images

def create_mosaic(class_images, n_rows=9, n_cols=30, images_per_class=30):
    """
    Create a mosaic grid with 9 rows (one per class) and 30 columns.
    Each row contains 30 randomly selected images from the corresponding class.
    """
    mosaic = Image.new('RGB', (n_cols * 128, n_rows * 128))
    for class_label in sorted(class_images.keys()):
        images = class_images[class_label]
        if len(images) < images_per_class:
            raise ValueError(f"Not enough images for class {class_label}. Found {len(images)}, need {images_per_class}.")
        selected_images = random.sample(images, images_per_class)
        for i, img in enumerate(selected_images):
            mosaic.paste(img, (i * 128, (class_label - 1) * 128))
    return mosaic

def main():
    folder = "/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_train"  # Replace with your folder path
    class_images = load_images_from_folder(folder)
    mosaic = create_mosaic(class_images)
    mosaic.save("mosaic_9x30.png")
    print("Mosaic saved as mosaic_9x30.png")

if __name__ == "__main__":
    main()
