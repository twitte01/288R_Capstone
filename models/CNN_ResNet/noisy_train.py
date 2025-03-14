import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset  # Import dataset
from model import create_model  # Import model
import torchvision.transforms as transforms
import os
import csv
import random
from pathlib import Path

# Define checkpoint path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "models" / "CNN_ResNet" / "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet.pth")
NOISY_CHECKPOINT_DIR = ROOT_DIR / "models" / "CNN_ResNet" / "checkpoints"
NOISY_CHECKPOINT_PATH = os.path.join(NOISY_CHECKPOINT_DIR, "noisy_resnet.pth")

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)

# Extract best parameters
best_params = checkpoint.get("params", {})

# Print the best parameters to verify
print("Using Best Parameters:", best_params)

# Define device (MPS or CPU or GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())
    

if __name__ == "__main__":  #  Prevent multiprocessing issues
    # Load dataset
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    noisy_data_dir = ROOT_DIR / "data" / "images" / "Speech Commands_noise"
    val_list_path = ROOT_DIR / "docs" / "validation_list.txt"
    train_list_path = ROOT_DIR / "docs" / "training_list.txt"
    val_filenames = load_filenames(val_list_path)
    train_filenames = load_filenames(train_list_path)

    # Define transforms (same as before)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = SpeechCommandsDataset(noisy_data_dir, file_list=train_filenames, transform=transform)
    val_dataset = SpeechCommandsDataset(noisy_data_dir, file_list=val_filenames, transform=transform)
    print(f"Dataset size: {len(train_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    model = create_model(num_classes=len(train_dataset.classes))
    model.to(device)


    def train_noisy_model(lr, batch_size, optimizer_name, epochs=10):
        """Train a fresh model on noisy data using best hyperparameters."""
        
        # Create DataLoaders with the given batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize model
        model = create_model(num_classes=len(train_dataset.classes))
        model.to(device)

        # Select optimizer based on best params
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Training history
        training_history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

            # Evaluate on validation set
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

            # Save training history for this epoch
            training_history.append([epoch + 1, running_loss, train_acc, val_loss, val_acc])

        print(f"Final Validation Accuracy: {val_acc:.2f}%")
        return model, optimizer, val_acc, training_history

    
    # Evaluation function
    def evaluate_model(model, val_loader, criterion):
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        return avg_val_loss, val_acc
    
    # Extract best hyperparameters
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    optimizer_name = best_params["optimizer"]

    # Train the fresh model on noisy data
    noisy_model, best_optimizer, val_acc, training_history = train_noisy_model(lr=lr, batch_size=batch_size, optimizer_name=optimizer_name, epochs=10)

    # Ensure checkpoint directory exists
    os.makedirs(NOISY_CHECKPOINT_DIR, exist_ok=True)

    # Save model with optimizer state
    torch.save({
        "model_state_dict": noisy_model.state_dict(),
        "optimizer_state_dict": best_optimizer.state_dict(),
        "params": best_params
    }, NOISY_CHECKPOINT_PATH)

    print(f"Trained noisy model saved at {NOISY_CHECKPOINT_PATH}")

    # Save training history for best model
    history_file = os.path.join(CHECKPOINT_DIR, "noisy_training_history.csv")
    with open(history_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
        writer.writerows(training_history)

    print(f"Best model training history saved to {history_file}")
