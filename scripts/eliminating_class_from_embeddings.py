import numpy as np
import argparse
import os

def filter_class(input_file, output_file, class_number):
    # Load npz file
    data = np.load(input_file, allow_pickle=True)
    
    # Assume keys: "embeddings" and "paths"
    embeddings = data["embeddings"]
    paths = data["image_paths"]

    # Filter: keep entries NOT matching the given class number
    mask = [f"_ {class_number}." not in os.path.basename(p).replace("class", "_") and 
            not os.path.basename(p).endswith(f"_{class_number}.jpg")
            for p in paths]

    filtered_embeddings = embeddings[mask]
    filtered_paths = paths[mask]

    # Save new npz file
    np.savez(output_file, embeddings=filtered_embeddings, image_paths=filtered_paths)
    print(f"Saved filtered file: {output_file}")
    print(f"Removed {len(paths) - len(filtered_paths)} samples of class {class_number}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out a class from embeddings npz file.")
    parser.add_argument("--input_file", required=True, help="Path to input .npz file")
    parser.add_argument("--class_number", type=int, required=True, help="Class number to remove")
    parser.add_argument("--output_file", required=True, help="Path to save filtered .npz file")
    
    args = parser.parse_args()
    filter_class(args.input_file, args.output_file, args.class_number)

