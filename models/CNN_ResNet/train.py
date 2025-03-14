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

# Define device (MPS or CPU or GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

if __name__ == "__main__":  #  Prevent multiprocessing issues
    # Load dataset
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    data_dir = ROOT_DIR / "data" / "images" / "Speech Commands (trimmed)"
    val_list_path = ROOT_DIR / "docs" / "validation_list.txt"
    train_list_path = ROOT_DIR / "docs" / "training_list.txt"
    val_filenames = load_filenames(val_list_path)
    train_filenames = load_filenames(train_list_path)

    print(f"Number of training files: {len(train_filenames)}")
    print(f"Number of validation files: {len(val_filenames)}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print(list(train_filenames)[:5])
    print(list(val_filenames)[:5])

    # Create datasets
    train_dataset = SpeechCommandsDataset(data_dir, file_list=train_filenames, transform=transform)
    val_dataset = SpeechCommandsDataset(data_dir, file_list=val_filenames, transform=transform)
    print(f"Dataset size: {len(train_dataset)}")

    print(train_dataset)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    model = create_model(num_classes=len(train_dataset.classes))
    model.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    CHECKPOINT_DIR = ROOT_DIR / "models" / "CNN_ResNet" / "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet.pth")
    # Check if a checkpoint exists
    start_epoch = 0  # Default to start from scratch
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
        print(f" Model loaded, resuming from epoch {start_epoch}")

    best_history = []

    # Training function with logging
    def train_model(lr, batch_size, optimizer_name, epochs=10):
        """Train the model with given hyperparameters and return validation accuracy."""

        # Create DataLoaders with the given batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Initialize model
        model = create_model(num_classes=len(train_dataset.classes))
        model.to(device)

        # Select optimizer based on given name
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
            running_loss = 0.0
            correct, total = 0, 0

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


    # Define the range of hyperparameters
    learning_rates = [1e-5, 1e-4, 5e-4, 1e-3]
    batch_sizes = [32, 64, 128]
    optimizers = ["adam", "sgd"]

    num_trials = 8  # Number of random experiments
    best_acc = 0.0
    best_params = {}
    results = []
    best_model = None  

    for trial in range(num_trials):
        # Randomly select hyperparameters
        lr = random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        optimizer_name = random.choice(optimizers)

        print(f"\n==== Trial {trial+1}/{num_trials} ====")
        print(f"Training with: LR={lr}, Batch Size={batch_size}, Optimizer={optimizer_name}")

        # Train and get validation accuracy
        model, optimizer, val_acc, training_history = train_model(lr=lr, batch_size=batch_size, optimizer_name=optimizer_name, epochs=10)

        # Save the result
        results.append([trial + 1, lr, batch_size, optimizer_name, val_acc])

        # Save the best configuration
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = {"lr": lr, "batch_size": batch_size, "optimizer": optimizer_name}
            
            # Save the best model instance
            best_model = model
            best_model_state = model.state_dict()
            # Save the optimizer state as well
            best_optimizer_state = optimizer.state_dict()
            best_training_history = training_history

    # Save best model properly
    if best_model is not None:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
        torch.save({
            "model_state_dict": best_model_state,
            "optimizer_state_dict": best_optimizer_state,  # Save optimizer state
            "params": best_params
        }, CHECKPOINT_PATH)
        print(f"Best model saved successfully at {CHECKPOINT_PATH}")

        # Save training history for best model
        history_file = os.path.join(CHECKPOINT_DIR, "training_history.csv")
        with open(history_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
            writer.writerows(best_training_history)

        print(f"Best model training history saved to {history_file}")

    else:
        print("No best model found.")


    
