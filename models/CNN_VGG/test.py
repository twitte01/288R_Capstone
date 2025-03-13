import torch
from model import create_model
from dataset import SpeechCommandsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, matthews_corrcoef

from team14.nifty_confusion_matrix import NiftyConfusionMatrix

# Define project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "CNN_VGG" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "CNN_VGG" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure results directory exists

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_tuned_model.pth"
RESULTS_PATH = RESULTS_DIR / "tuned_best_model_test_results.txt"
CONF_MATRIX_PATH = RESULTS_DIR / "tuned_best_model_confusion_matrix.png"
CONF_MATRIX_CSV = RESULTS_DIR / "tuned_best_model_confusion_matrix.csv"

# Define device (MPS for Mac, CUDA for Nvidia, CPU as fallback)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths relative to project root
data_dir = PROJECT_ROOT.parent / "data/images/Speech Commands (trimmed)"
test_list_path = PROJECT_ROOT.parent / "docs" / "testing_list.txt"

# Load dataset filenames
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
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Load trained model
    model = create_model(num_classes=len(test_dataset.classes))

    # Ensure the checkpoint file exists before loading
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        #  Checkpoint contains optimizer and metadata
         model.load_state_dict(checkpoint["model_state_dict"])
         print(f"Model loaded from {CHECKPOINT_PATH} (full checkpoint)")
        else:
            # Checkpoint only contains model weights
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {CHECKPOINT_PATH} (weights only)")
    else:
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        exit()

    model.to(device)
    model.eval()

    # Evaluate model and compute metrics
    def evaluate_model(model, test_loader, class_names):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute evaluation metrics
        test_acc = accuracy_score(all_labels, all_preds) * 100
        cohen_kappa = cohen_kappa_score(all_labels, all_preds)
        mcc_score = matthews_corrcoef(all_labels, all_preds)

        print(f"\nTest Accuracy: {test_acc:.2f}%")
        print(f"Cohen’s Kappa Score: {cohen_kappa:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}")

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

        # Generate classification report
        class_report = classification_report(all_labels, all_preds, target_names=class_names)

        # Save results to file
        with open(RESULTS_PATH, "w") as f:
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Cohen’s Kappa Score: {cohen_kappa:.4f}\n")
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc_score:.4f}\n\n")
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

        print(f"\nTest results saved to {RESULTS_PATH}")

        # Create a NiftyConfusionMatrix object
        my_cm = NiftyConfusionMatrix(cm, class_names)

        # Display and save confusion matrix
        my_cm.display()
        plt.savefig(CONF_MATRIX_PATH)  # Save confusion matrix as an image
        print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

        # Save confusion matrix to CSV
        my_cm.to_csv()
        print(f"Confusion matrix saved to {CONF_MATRIX_CSV}")

    # Run evaluation
    evaluate_model(model, test_loader, test_dataset.classes)