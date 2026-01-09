import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def extract_class_from_path(path):
    """Extract class from image path (last digit before .jpg)"""
    filename = path.split('/')[-1]
    class_str = filename.split('.')[0].split('_')[-1]
    return int(class_str)

def calculate_median_embeddings(embeddings, classes):
    """Calculate median embedding for each class"""
    unique_classes = np.unique(classes)
    median_embeddings = {cls: np.median(embeddings[classes == cls], axis=0) for cls in unique_classes}
    return median_embeddings, unique_classes

def find_closest_median(embedding, median_embeddings):
    """Find the closest median for a given embedding"""
    distances = {cls: euclidean(embedding, median) for cls, median in median_embeddings.items()}
    min_distance = min(distances.values())
    closest_classes = [cls for cls, dist in distances.items() if dist == min_distance]
    return closest_classes

def analyze_embeddings(npz_path):
    """Analyze embeddings and print a summary of classification results"""
    data = np.load(npz_path)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    classes = np.array([extract_class_from_path(path) for path in image_paths])

    median_embeddings, unique_classes = calculate_median_embeddings(embeddings, classes)

    # Initialize results dictionary
    results = {cls: {other_cls: 0 for other_cls in unique_classes} for cls in unique_classes}
    ties = {}

    for emb, cls in zip(embeddings, classes):
        closest_classes = find_closest_median(emb, median_embeddings)
        if len(closest_classes) > 1:
            tie_key = tuple(sorted(closest_classes))
            ties[tie_key] = ties.get(tie_key, 0) + 1
        else:
            closest_class = closest_classes[0]
            results[cls][closest_class] += 1

    # Print summary
    for cls in sorted(unique_classes):
        correctly_classified = results[cls][cls]
        incorrectly_classified = sum(results[cls][other_cls] for other_cls in unique_classes if other_cls != cls)

        print(f"Class {cls}:")
        print(f"- Correctly classified: {correctly_classified}")
        print(f"- Incorrectly classified: {incorrectly_classified}")
        for other_cls in sorted(unique_classes):
            if other_cls != cls and results[cls][other_cls] > 0:
                print(f"  - In class {other_cls}: {results[cls][other_cls]}")
        print()

# Example usage:
analyze_embeddings('../embeddings_files/resnet_embeddings_train.npz')
