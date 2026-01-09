import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    """Analyze embeddings and return DataFrames with the results"""
    data = np.load(npz_path)
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    classes = np.array([extract_class_from_path(path) for path in image_paths])
    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Total image paths: {len(image_paths)}")
    print(f"Unique classes: {np.unique(classes)}")
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

    # Prepare DataFrame for results
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'True Class'
    df_results.columns.name = 'Closest Median Class'

    # Add total instances per class
    df_results['Total'] = df_results.sum(axis=1)

    # Calculate % correct
    df_results['% Correct'] = (df_results.apply(lambda row: row[row.name], axis=1) / df_results['Total']) * 100

    # Normalized confusion matrix
    df_normalized = df_results[unique_classes].copy().astype(float)
    for cls in unique_classes:
        df_normalized.loc[cls, :] = df_results.loc[cls, unique_classes] / df_results.loc[cls, 'Total'] * 100

    # Prepare DataFrame for ties
    df_ties = pd.DataFrame(list(ties.items()), columns=['Tied Classes', 'Count'])

    return df_results, df_normalized, df_ties, unique_classes

def plot_confusion_matrix(df, unique_classes, title, is_normalized=False):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    if is_normalized:
        sns.heatmap(df[unique_classes], annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': '%'})
    else:
        sns.heatmap(df[unique_classes], annot=True, fmt="d", cmap="Reds")
    plt.title(title)
    plt.xlabel("Closest Median Class")
    plt.ylabel("True Class")
    plt.show()




# Example usage:
df_results, df_normalized, df_ties, unique_classes = analyze_embeddings('../embeddings_files/resnet_embeddings_train.npz')
df_results.to_csv('../analysis_of_embedding_distance/embedding_analysis_resnet.csv')
df_normalized.to_csv('../analysis_of_embedding_distance/embedding_analysis_normalized_resnet.csv')
df_ties.to_csv('../analysis_of_embedding_distance/embedding_ties_resnet.csv')
plot_confusion_matrix(df_results, unique_classes, "Confusion Matrix For resnet: Closest Median Class (Counts)")
plot_confusion_matrix(df_normalized, unique_classes, "Confusion Matrix For resnet: Closest Median Class (Normalized %)", is_normalized=True)
