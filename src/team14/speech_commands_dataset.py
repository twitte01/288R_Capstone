from torch.utils.data import Dataset
from PIL import Image
import os

class SpeechCommandsDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(set(f.split('/')[0] for f in file_list)))}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        label_name = self.file_list[idx].split('/')[0]
        label = self.class_to_idx[label_name]

        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(self.file_list[idx])

        return image, label, filename