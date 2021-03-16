import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from .ingest import ingest_fer13


class FER2013(data.Dataset):

    def __init__(self, split='Training', train_data=None, train_labels=None, publicTest_data=None, publicTest_labels=None, privateTest_data=None, privateTest_labels=None):
        # split can be 'Training', 'PublicTest', or 'PrivateTest'
        self.split = split
        # now load the arrays
        self.train_data = train_data
        self.train_labels = train_labels
        self.publicTest_data = publicTest_data
        self.publicTest_labels = publicTest_labels
        self.privateTest_data = privateTest_data
        self.privateTest_labels = privateTest_labels

        self.train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.test_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])


    def __getitem__(self, index):
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.publicTest_data[index], self.publicTest_labels[index]
        else:
            img, target = self.privateTest_data[index], self.privateTest_labels[index]

        img = Image.fromarray(img.astype('uint8'))
        if self.split == 'Training':
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.publicTest_data)
        else:
            return len(self.privateTest_data)