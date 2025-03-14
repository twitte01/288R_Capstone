import torch
from model import create_model
from dataset import SpeechCommandsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Define device (MPS for Mac, CUDA for Nvidia, CPU as fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
data_dir = ROOT_DIR / "data" / "images" / "Speech Commands (trimmed)"
noise_data_dir = ROOT_DIR / "data" / "images" / "Speech Commandss (noise)"
test_list_path = "docs/testing_list.txt"

# Get Class Labels
classes = []
for folder in os.listdir(data_dir):
    classes.append(folder)

def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

if __name__ == "__main__":  # Prevent multiprocessing issues
    test_filenames = load_filenames(test_list_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = SpeechCommandsDataset(data_dir, file_list=test_filenames, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    noise_test_dataset = SpeechCommandsDataset(noise_data_dir, file_list=test_filenames, transform=transform)
    noise_test_loader = DataLoader(noise_test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


    # Load trained model
    CHECKPOINT_DIR = ROOT_DIR / "models" / "CNN_ResNet" / "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_resnet.pth")
    RESULTS_DIR = ROOT_DIR / "models" / "CNN_ResNet" / "results"
    RESULTS_PATH = os.path.join(RESULTS_DIR, "test_results.txt")
    CONF_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    CONF_MATRIX_CSV_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    NOISE_RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_test_results.txt")
    NOISE_CONF_MATRIX_PATH = os.path.join(RESULTS_DIR, "noise_confusion_matrix.png")
    NOISE_CONF_MATRIX_CSV_PATH = os.path.join(RESULTS_DIR, "noise_confusion_matrix.csv")


    model = create_model(num_classes=len(test_dataset.classes))

    # Ensure the checkpoint file exists before loading
    # Load the full checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    
    if os.path.exists(CHECKPOINT_PATH):
        # Load the model state dictionary correctly
        model.load_state_dict(checkpoint["model_state_dict"]) 
        print(f"Model loaded from {CHECKPOINT_PATH}")
    else:
        print(f" Error: Checkpoint not found at {CHECKPOINT_PATH}")
        exit()

    model.to(device)
    model.eval()

    # Evaluate model and generate confusion matrix
    def evaluate_model(model, test_loader, class_names, RESULTS_PATH, CONF_MATRIX_PATH, CONF_MATRIX_CSV_PATH):
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        
        # Generate classification report
        class_report = classification_report(all_labels, all_preds, target_names=class_names)

        # Save results to file
        # Save results to file
        with open(RESULTS_PATH, "w") as f:
            # f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
            # f.write("Classification Report:\n")
            # f.write(class_report)
            # f.write("\nConfusion Matrix:\n")
            # np.savetxt(f, cm, fmt="%d")
            # f.write("\nNormalized Confusion Matrix:\n")
            # np.savetxt(f, cm_normalized, fmt="%.2f")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(class_report)
            f.write("\nConfusion Matrix (Raw Counts):\n")

            # Add headers (column labels)
            f.write(" " * 12 + " ".join(f"{name[:6]:>6}" for name in class_names) + "\n")
            f.write("-" * (12 + len(class_names) * 7) + "\n")

            # Add row labels and data
            for i, row in enumerate(cm):
                f.write(f"{class_names[i][:10]:<10} | " + " ".join(f"{int(val):>6}" for val in row) + "\n")

            f.write("\nConfusion Matrix (Normalized):\n")

            # Add headers (column labels)
            f.write(" " * 12 + " ".join(f"{name[:6]:>6}" for name in class_names) + "\n")
            f.write("-" * (12 + len(class_names) * 7) + "\n")

            # Add row labels and normalized data
            for i, row in enumerate(cm_normalized):
                f.write(f"{class_names[i][:10]:<10} | " + " ".join(f"{val:.2f}" for val in row) + "\n")


        print(f"Test results saved to {RESULTS_PATH}")

        # Plot Confusion Matrix
        plt.figure(figsize=(12, 10))  
        ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                         xticklabels=class_names, yticklabels=class_names, linewidths=0.5,
                         annot_kws={"size": 10})  

        plt.xticks(rotation=45, ha="right", fontsize=12) 
        plt.yticks(rotation=0, fontsize=12)
        plt.xlabel("Predicted Labels", fontsize=14, labelpad=15)
        plt.ylabel("True Labels", fontsize=14, labelpad=15)
        plt.title("Confusion Matrix (Normalized)", fontsize=16, pad=15)
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PATH)
        print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")
        plt.show()

        # Export Confusion Matrix to csv 
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(CONF_MATRIX_CSV_PATH)
        print(f"Confusion matrix saved to {CONF_MATRIX_CSV_PATH}")

        
    # Run evaluation
    evaluate_model(model, test_loader, test_dataset.classes, RESULTS_PATH, CONF_MATRIX_PATH, CONF_MATRIX_CSV_PATH)

    # Run noisy evaluation
    evaluate_model(model, noise_test_loader, test_dataset.classes, NOISE_RESULTS_PATH, NOISE_CONF_MATRIX_PATH, NOISE_CONF_MATRIX_CSV_PATH)


