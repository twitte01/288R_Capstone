import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import itertools
import csv
from pathlib import Path
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef

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

train_filenames = load_filenames(train_list_path)
val_filenames = load_filenames(val_list_path)

#  Define device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
batch_size = 16
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


# Function to handle model freezing
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

def load_checkpoint(model, optimizer, scheduler, config):
    """Load checkpoint for a specific hyperparameter configuration if it exists."""
    lr, opt, weight_decay, freeze_layers = config
    checkpoint_name = f"model_lr={lr}_opt={opt}_wd={weight_decay}_freeze={freeze_layers}.pth"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name

    start_epoch = 0
    best_val_loss = float("inf")

    if checkpoint_path.exists():
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss
    
 # Track best model
best_model_state = None  
best_model_val_acc = 0.0 

# Function to train and evaluate the model
def train_and_evaluate(config, model, train_loader, val_loader):
    global best_model_state, best_model_val_acc
    
    lr, opt, weight_decay, freeze_layers = config
    model.to(device)
    fine_tune_model(model, freeze_layers)

    # ✅ Ensure model is in train mode
    model.train()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if opt == "Adam" else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler() if torch.cuda.is_available() else None

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, config)
    patience = 3
    early_stop_counter = 0
    print_every = 100 

    for epoch in range(start_epoch, 10):
        model.train()  # ✅ Ensure train mode before training loop
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # ✅ Ensure loss is computed with gradients enabled
            if torch.cuda.is_available():
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Train Acc: {100 * correct / total:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ✅ Validation Step
        model.eval()  # Switch to evaluation mode
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        scheduler.step(avg_val_loss)

        # ✅ Compute additional evaluation metrics
        cohen_kappa = cohen_kappa_score(all_labels, all_preds)
        mcc_score = matthews_corrcoef(all_labels, all_preds)

        print(f"Epoch {epoch+1}/10 - Config: {config} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Kappa: {cohen_kappa:.4f}, MCC: {mcc_score:.4f}")

        # ✅ If this model has the best validation accuracy, save it separately
        if val_acc > best_model_val_acc:
            best_model_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"🔹 New Best Model Found! Saving with Validation Accuracy: {val_acc:.2f}%") 

    # ✅ Save final model checkpoint for this configuration
    checkpoint_name = f"model_lr={lr}_opt={opt}_wd={weight_decay}_freeze={freeze_layers}.pth"
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)

    return train_loss, train_acc, avg_val_loss, val_acc, cohen_kappa, mcc_score, best_model_state


best_config = None
best_accuracy = 0.0
best_model_state = None
results = []

for i, config in enumerate(random_configs):
    print(f"\nTesting Configuration {i+1}/{len(random_configs)}: {config}")
    model = create_model(num_classes=len(train_dataset.classes))
    train_loss, train_acc, val_loss, val_acc, cohen_kappa, mcc_score, model_state = train_and_evaluate(config, model, train_loader, val_loader)
    results.append((config, train_loss, train_acc, val_loss, val_acc, cohen_kappa, mcc_score))

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_config = config
        best_model_state = model_state

## ✅ Save the best model at the end of training
if best_model_state is not None:
    best_model_path = CHECKPOINT_DIR / "best_tuned_model.pth"
    torch.save({
        "model_state_dict": best_model_state,
        "best_val_acc": best_accuracy  # ✅ Save best validation accuracy too
    }, best_model_path)
    print(f"✅ Best model saved with Validation Accuracy: {best_accuracy:.2f}% at {best_model_path}")
    
# Save results
history_file = PROJECT_ROOT / "CNN_VGG" / "results" / "hyperparameter_tuning_results.csv"
with open(history_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Learning Rate", "Optimizer", "Weight Decay", "Freeze Layers", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Cohen Kappa", "MCC Score"])
    for config, train_loss, train_acc, val_loss, val_acc, cohen_kappa, mcc_score in results:
        writer.writerow([*config, train_loss, train_acc, val_loss, val_acc, cohen_kappa, mcc_score])

print(f"\n Best Configuration: {best_config} with Accuracy: {best_accuracy:.2f}%")