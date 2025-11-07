import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

from torchvision import models
import torch.nn as nn


# Define preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CustomEmbedder(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()

        # Rebuild the same architecture used in training
        model = models.resnet50(weights=None)

        # This MUST match how you fine-tuned your model
        model.fc = nn.Linear(2048, 9)  # Adjust to the actual number of classes used during training

        # Load the weights now that the fc layer matches
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # Remove final classification layer to get 2048D features
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # [B, 2048, 1, 1]

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features.view(features.size(0), -1)  # [B, 2048]



device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = None  # Initialized in main()

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    features = model(image)
    return features.cpu().numpy()

def extract_embeddings_from_directory(directory_path):
    embeddings = []
    image_paths = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            print(f"📸 Processing {filename}...")
            features = extract_features(image_path)
            embeddings.append(features)
            image_paths.append(image_path)

    embeddings = np.vstack(embeddings)
    return embeddings, image_paths

def save_embeddings(embeddings, image_paths, output_file):
    np.savez(output_file, embeddings=embeddings, image_paths=image_paths)
    print(f"💾 Saved embeddings to {output_file}")

def main():
    global model

    parser = argparse.ArgumentParser(description="Extract 2048D embeddings using a custom model.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file (.npz).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to your pre-trained model (.pth).")
    args = parser.parse_args()

    model = CustomEmbedder(args.model_path).to(device).eval()
    print(f"✅ Loaded custom model from {args.model_path}")
    print(f"✅ Using device: {device}")

    embeddings, image_paths = extract_embeddings_from_directory(args.image_dir)
    save_embeddings(embeddings, image_paths, args.output_file)

if __name__ == "__main__":
    main()
