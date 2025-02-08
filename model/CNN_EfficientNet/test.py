import torch
from model import create_model
from dataset import SpeechCommandsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define device (MPS or CPU or GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data_dir = "data/images/Speech Commands (trimmed)"
test_list_path = "testing_list.txt"


def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

if __name__ == "__main__":  # <-- Add this guard
    # Load dataset
    data_dir = "data/images/Speech Commands(trimmed)"
    test_list_path = "path_to_testing_list.txt"
    test_filenames = load_filenames(test_list_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = SpeechCommandsDataset(data_dir, file_list=test_filenames, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Load trained model
    model = create_model(num_classes=len(test_dataset.classes))
    model.load_state_dict(torch.load("checkpoints/efficientnet_speech_commands.pth"))
    model.to(device)
    model.eval()

    # Evaluate model
    def evaluate_model(model, val_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")