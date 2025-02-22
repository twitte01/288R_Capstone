import torch
from models.CNN_EfficientNet.dataset import SpeechCommandsDataset  # Import the dataset class
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

data_dir = "data/images/Speech Commands (trimmed)"
val_list_path = "docs/validation_list.txt"
train_list_path = "docs/training_list.txt"

# Load the validation and test filenames from the provided .txt files
def load_filenames(file_path):
    with open(file_path, "r") as f:
        return set(line.strip().replace(".wav", ".png") for line in f.readlines())

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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
