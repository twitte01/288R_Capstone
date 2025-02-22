import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpeechCommandsDataset  # Import dataset
from model import create_model  # Import model
import torchvision.transforms as transforms
import os
import csv

# Define device (MPS or CPU or GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

if __name__ == "__main__":  #  Prevent multiprocessing issues
    # Load dataset
    data_dir = "data/images/Speech Commands (trimmed)"
    val_list_path = "docs/validation_list.txt"
    train_list_path = "docs/training_list.txt"
    val_filenames = load_filenames(val_list_path)
    train_filenames = load_filenames(train_list_path)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = SpeechCommandsDataset(data_dir, file_list=train_filenames, transform=transform)
    val_dataset = SpeechCommandsDataset(data_dir, file_list=val_filenames, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    model = create_model(num_classes=len(train_dataset.classes))
    model.to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    CHECKPOINT_DIR = "/Users/taylorwitte/Documents/288R_Capstone/288R_Capstone/models/CNN_ResNet/checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    # Check if a checkpoint exists
    start_epoch = 0  # Default to start from scratch
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
        print(f" Model loaded, resuming from epoch {start_epoch}")

    # Training function with logging
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3, print_every=10, start_epoch=0):
        best_val_loss = float("inf")
        early_stop_counter = 0
        history = []  # List to store epoch-wise results

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            # Get current learning rate
            lr = optimizer.param_groups[0]['lr']

            for batch_idx, (images, labels) in enumerate(train_loader):
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

                if (batch_idx + 1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

            print(f"Epoch [{epoch+1}/{epochs}] Complete, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Learning Rate: {lr:.6f}")

            # Store epoch results
            history.append([epoch+1, train_loss, train_acc, val_loss, val_acc, lr])

            # Ensure checkpoint directory exists
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss
                }
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                print("Model improved and saved!")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter}/{patience} epochs.")

            if early_stop_counter >= patience:
                print("Early stopping triggered! Stopping training.")
                break

        # Save history to CSV
        history_file = os.path.join(CHECKPOINT_DIR, "training_history.csv")
        with open(history_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Learning Rate"])
            writer.writerows(history)
        print(f"Training history saved to {history_file}")

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

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, start_epoch=start_epoch)


    # Save model 
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "resnet_speech_commands.pth")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("Model saved successfully!")
