import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import itertools
import os
import csv
from dataset import SpeechCommandsDataset
from model import create_model
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Checkpoint Directory
CHECKPOINT_DIR = "models/CNN_EfficientNet/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model2.pth")

# Load dataset
data_dir = "data/images/Speech Commands (trimmed)"
val_list_path = "docs/validation_list.txt"
train_list_path = "docs/training_list.txt"

def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

train_filenames = load_filenames(train_list_path)
val_filenames = load_filenames(val_list_path)

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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Define hyperparameter search space
hyperparams = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "optimizer": ["Adam", "SGD"],
    "weight_decay": [0, 1e-4, 1e-5],
    "freeze_layers": [True, False]
}

# Randomly sample hyperparameter configurations
num_configs = 3
# Generate random hyperparameter combinations properly
all_configs = list(itertools.product(
    hyperparams["learning_rate"],
    hyperparams["optimizer"],
    hyperparams["weight_decay"],
    hyperparams["freeze_layers"]
))
random_configs = random.sample(all_configs, num_configs)  # Select N random configurations
#random_configs = [(0.0005, 'Adam', 0, False)]

def fine_tune_model(model, freeze_layers=False):
    """Freeze or gradually unfreeze EfficientNet feature extractor."""
    if freeze_layers:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Feature extractor frozen.")
    else:
        for param in model.features.parameters():
            param.requires_grad = True
        print("Feature extractor trainable.")

def load_checkpoint(model, optimizer, scheduler, opt, lr, weight_decay, freeze_layers):
    """Load checkpoint if exists, ensuring each configuration is separate."""
    start_epoch = 0
    best_val_loss = float("inf")

    # Create a unique checkpoint file for this configuration
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"best_model_{opt}_lr{lr}_wd{weight_decay}_freeze{freeze_layers}.pth"
    )

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        saved_optimizer_type = checkpoint.get("optimizer_type", None)
        if saved_optimizer_type != opt:
            print(f"Warning: Checkpoint optimizer was {saved_optimizer_type}, but current optimizer is {opt}. Reinitializing optimizer.")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss, checkpoint_path

def train_and_evaluate(config, model, train_loader, val_loader):
    lr, opt, weight_decay, freeze_layers = config
    model.to(device)
    fine_tune_model(model, freeze_layers)

    criterion = nn.NLLLoss()

    # Unique filename based on the hyperparameter configuration
    CHECKPOINT_PATH = os.path.join(
        CHECKPOINT_DIR, f"best_model_{opt}_lr{lr}_wd{weight_decay}_freeze{freeze_layers}.pth"
    )

    # Select optimizer
    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if opt == "Adam" else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Fix: Remove `verbose=True` (deprecated)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Fix: Enable GradScaler only if CUDA is available
    scaler = GradScaler() if torch.cuda.is_available() else None

    start_epoch, best_val_loss, CHECKPOINT_PATH = load_checkpoint(model, optimizer, scheduler, opt, lr, weight_decay, freeze_layers)
    patience = 3
    early_stop_counter = 0
    print_every = 100  # Print progress every 100 batches

    for epoch in range(start_epoch, 10):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Fix: Use `autocast()` only if CUDA is available
            if torch.cuda.is_available():
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Fix: Use `scaler` only if CUDA is available
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Print progress every `print_every` batches
            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Train Acc: {100 * correct / total:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation Step
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:  # Take only images & labels
                images, labels = batch[:2]
                images, labels = images.to(device), labels.to(device)

                if torch.cuda.is_available():
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Fix: Print updated learning rate (instead of using `verbose=True`)
        scheduler.step(avg_val_loss)
        print(f"Learning rate updated: {scheduler.get_last_lr()}")

        print(f"Epoch {epoch+1}/10 - Config: {config} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer_type": opt
            }, CHECKPOINT_PATH)
            print("Model improved and saved to {CHECKPOINT_PATH}!")

        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break
    return train_loss, train_acc, avg_val_loss, val_acc

best_config = None
best_accuracy = 0.0
results = []

for i, config in enumerate(random_configs):
    print(f"\nTesting Configuration {i+1}/{len(random_configs)}: {config}")
    model = create_model(num_classes=len(train_dataset.classes))
    model.to(device)
    train_loss, train_acc, val_loss, val_acc = train_and_evaluate(config, model, train_loader, val_loader)
    results.append((config, train_loss, train_acc, val_loss, val_acc))

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_config = config

history_file = "hyperparameter_tuning_results.csv"
with open(history_file, mode="w", newline="") as f: #switch mode to a to write to current doc
    writer = csv.writer(f)
    writer.writerow(["Learning Rate", "Optimizer", "Weight Decay", "Freeze Layers", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
    for config, train_loss, train_acc, val_loss, val_acc in results:
        writer.writerow([*config, train_loss, train_acc, val_loss, val_acc])

print(f"\nBest Configuration: {best_config} with Accuracy: {best_accuracy:.2f}%")
