import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Utility functions for data loading and preprocessing

def load_data(batch_size):
    # Load MNIST dataset as an example
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader

def preprocess_data(data):
    # Flatten the image data
    processed_data = data.view(data.size(0), -1)
    return processed_data

# Example usage:
# Replace this with your actual data loading and preprocessing logic
batch_size = 64
train_loader = load_data(batch_size)

# Iterate through the data loader to get a batch for demonstration
for images, _ in train_loader:
    processed_data = preprocess_data(images)
    print("Original Data Shape:", images.shape)
    print("Processed Data Shape:", processed_data.shape)
    break
