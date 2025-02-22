import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, file_list=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != "_background_noise_"])

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".png"):
                    file_path = os.path.join(class_name, filename)  # Relative path
                    if file_list is None or file_path in file_list:
                        self.data.append((file_path, self.classes.index(class_name)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
