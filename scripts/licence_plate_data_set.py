# dataset/license_plate_dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
try:
    from utils.charset import char2idx
except ImportError:
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
import torchvision.transforms as transforms

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, label_file, img_size=(100, 32)):
        self.image_dir = image_dir
        self.img_size = img_size
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                self.samples.append((filename, label))

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C x H x W]
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # Load grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.img_size)

        image = self.transform(image)  # [1 x 32 x 100]
        
        # Convert label to indices
        label_indices = [char2idx[char] for char in label]

        return image, torch.tensor(label_indices, dtype=torch.long), len(label_indices)
