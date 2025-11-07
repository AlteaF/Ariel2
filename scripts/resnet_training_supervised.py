
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import os

def main():
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Load the data
    train_data = torchvision.datasets.ImageFolder(root="/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_resnet_baseline/cropped_train", transform=transform)
    test_data = torchvision.datasets.ImageFolder(root="/Users/alteafogh/Documents/ITU/Research_project/Finding_A_Nemo/dataset/cropped/cropped_resnet_baseline/cropped_test", transform=transform)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # Define the model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Freeze the first 3 stages (conv1, conv2_x, conv3_x)
    for name, param in model.named_parameters():
        # Freeze conv1, conv2_x, and conv3_x
        if name.startswith('layer1') or name.startswith('layer2') or name.startswith('conv1') or name.startswith('bn1'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    for name, layer in model.named_children():
        if name in ['conv1', 'bn1', 'layer1', 'layer2']:
            for param in layer.parameters():
                param.requires_grad = False

    # Replace the last layer for 9 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 9)  # 9 classes

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the parameters that require gradients
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    # Move the model to the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Define the number of epochs
    num_epochs = 10

    # Train the model
    for epoch in range(num_epochs):
        # Train the model on the training set
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update the training loss
            train_loss += loss.item() * inputs.size(0)

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                # Move the data to the device
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Update the test loss and accuracy
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == labels.data)

        # Save the model
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/resnet50_finetuned_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")

        # Print the training and test loss and accuracy
        train_loss /= len(train_data)
        test_loss /= len(test_data)
        test_acc = test_acc.float() / len(test_data)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
