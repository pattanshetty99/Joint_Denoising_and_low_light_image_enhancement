import os
import cv2
import torch
from torch.utils.data import Dataset


class LowLightDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = cv2.imread(os.path.join(self.lr_dir, self.lr_images[idx]))
        hr = cv2.imread(os.path.join(self.hr_dir, self.hr_images[idx]))

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB) / 255.0
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB) / 255.0

        lr = torch.tensor(lr).permute(2, 0, 1).float()
        hr = torch.tensor(hr).permute(2, 0, 1).float()

        return lr, hr
