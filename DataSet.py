from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class CelebA(Dataset):
    def __init__(self, img_size, path):
        self.img_size = img_size
        self.path = path
        self.names = os.listdir(path)
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.path, name))
        img_tensor = self.transform(img)
        return img_tensor
