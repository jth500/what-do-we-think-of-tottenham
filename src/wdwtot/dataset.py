import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class LogoDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def show_image(self, idx):
        img = self[idx][0]
        plt.imshow(img.permute(1, 2, 0))
