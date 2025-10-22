import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import argparse

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet-50 and remove the final classification layer
class ResNet50Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the 'fc' layer

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)  # Output shape: [B, 2048, 1, 1]
            features = features.view(features.size(0), -1)  # Flatten to [B, 2048]
        return features

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet50Embedder().to(device).eval()
print(f"✅ Using device: {device}")

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
    parser = argparse.ArgumentParser(description="Extract 2048D embeddings using ResNet-50.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file (.npz).")
    args = parser.parse_args()

    embeddings, image_paths = extract_embeddings_from_directory(args.image_dir)
    save_embeddings(embeddings, image_paths, args.output_file)

if __name__ == "__main__":
    main()

