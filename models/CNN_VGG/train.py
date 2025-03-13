import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import torchvision.transforms as transforms
import os

from dataset import SpeechCommandsDataset
from model import create_model

# Automatically detect project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Moves up to Capstone root
CHECKPOINT_DIR = PROJECT_ROOT / "CNN_VGG" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


# Load dataset
data_dir = PROJECT_ROOT.parent / "data/images/Speech Commands (trimmed)"
train_list_path = PROJECT_ROOT.parent / "docs" / "training_list.txt"
val_list_path = PROJECT_ROOT.parent / "docs" / "validation_list.txt"


def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

if __name__ == "__main__":  
    val_filenames = load_filenames(val_list_path)
    train_filenames = load_filenames(train_list_path)

  # Ensure train_filenames uses forward slashes and is trimmed of extra spaces
    train_filenames = {p.replace("\\", "/").strip() for p in train_filenames}
    val_filenames = {p.replace("\\", "/").strip() for p in val_filenames}

    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    
    # Create datasets
    train_dataset = SpeechCommandsDataset(str(data_dir), file_list=train_filenames, transform=transform)
    val_dataset = SpeechCommandsDataset(str(data_dir), file_list=val_filenames, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = create_model(num_classes=len(train_dataset.classes))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Paths for saving models
    BEST_MODEL_PATH = CHECKPOINT_DIR / "VGG_model.pth"
    LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pth"

    # Function to save a checkpoint
    def save_checkpoint(model, optimizer, epoch, loss, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")

    # Function to load checkpoint
    def load_checkpoint(model, optimizer, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            best_val_loss = checkpoint['loss']
            print(f"Resuming training from epoch {start_epoch}")
            return model, optimizer, start_epoch, best_val_loss
        return model, optimizer, 0, float("inf")

    # Load last checkpoint if it exists
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, LAST_CHECKPOINT_PATH)

    # Training function with periodic checkpoint saving
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3, print_every=10):
        global best_val_loss
        early_stop_counter = 0

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

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

            train_acc = 100 * correct / total
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            print(f"Epoch [{epoch+1}/{epochs}] Complete, Avg Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")

            # Save the last checkpoint (can be used to resume training)
            save_checkpoint(model, optimizer, epoch, val_loss, LAST_CHECKPOINT_PATH)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_checkpoint(model, optimizer, epoch, val_loss, BEST_MODEL_PATH)
                print("New best model saved!")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter}/{patience} epochs.")

            if early_stop_counter >= patience:
                print("Early stopping triggered! Stopping training.")
                break

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
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

