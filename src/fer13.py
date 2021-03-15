import numpy as np
import torch.utils.data as data
from ingest import ingest_fer13

class FER2013(data.Dataset):
    """`FER2013 Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', train_data=None, train_labels=None, publicTest_data=None, publicTest_labels=None, privateTest_data=None, privateTest_labels=None):
        # split can be 'Training', 'PublicTest', or 'PrivateTest'
        self.split = split
        self.transform = transform
        # now load the arrays
        self.train_data = train_data
        self.train_labels = train_labels
        self.publicTest_data = publicTest_data
        self.publicTest_labels = publicTest_labels
        self.privateTest_data = privateTest_data
        self.privateTest_labels = privateTest_labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.publicTest_data[index], self.publicTest_labels[index]
        else:
            img, target = self.privateTest_data[index], self.privateTest_labels[index]
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.publicTest_data)
        else:
            return len(self.privateTest_data)