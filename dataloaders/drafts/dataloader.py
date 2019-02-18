from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def show_sample(sample):
    """Show image with labels"""
    img, lbl = sample['image'], sample['label']
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img)
    fig.add_subplot(1,2,2)
    plt.imshow(lbl)
    plt.pause(0.001)  # pause a bit so that plots are updated


class OSMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.files = 0 # initialize empty

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_name, label_name = self.files[idx]
        image = io.imread(img_name)
        labels = io.imread(label_name)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def findallimages(self):
        subfolders = [f.path for f in os.scandir(self.root_dir) if f.is_dir()]
        images = []
        labels =[]
        for folder in subfolders:
            images += glob.glob(folder + "/*image.png")
            labels += glob.glob(folder + "/*labels.png")
        self.files = [i for i in zip(images, labels)]




    # Let’s instantiate this class and iterate through the data samples. We will print the sizes of first 4 samples and show their landmarks.

face_dataset =OSMDataset(root_dir='D:/programming/datasets/CITY-OSM/')
face_dataset.findallimages()
for i in range(3):
    sample = face_dataset[i]
    show_sample(sample)
    plt.show()


