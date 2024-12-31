import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from wdwtot.model.cnn import SimpleCNN

if __name__ == "__main__":

    dataset = ImageFolder(
        root="/Users/toby/Dev/what-do-we-think-of-tottenham/data/processed",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )

    # split into test and train
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # # Create a DataLoader for batch processing
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 4

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        # Training loop
        for inputs, labels in train_dataloader:

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_train_loss = running_loss / len(train_dataloader)

        # Validation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradient computation during validation
            for inputs, labels in test_dataloader:

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_dataloader)
        accuracy = 100 * correct / total

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Time for Epoch: {time.time() - start_time:.2f}s")
