from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import glob
from skimage import io, transform
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


class OSMDataset():
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
        self.ids = 0

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
        subfolders_names = [f.name for f in os.scandir(self.root_dir) if f.is_dir()]
        images = []
        labels =[]
        ids =[]

        for counter, folder in enumerate(subfolders):
            images += glob.glob(folder + "/*image.png")
            labels += glob.glob(folder + "/*labels.png")
            for i in range(len(glob.glob(folder + "/*image.png"))):
                ids.append(subfolders_names[counter])

        self.files = [i for i in zip(images, labels)]
        self.ids =  ids




    def divideOnTestandTrain(self, val_city_name, test_city_name):
        ''' returns testing and training images, depending on training, validation and test sets '''
        train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = [], [], [], [], [], []
        for i in range(len(self.files)):
            sample = self[i]
            city = self.ids[i]
            img, lbl = sample['image'], sample['label']
            if city == val_city_name:
                val_img.append(img)
                val_lbl.append(lbl)
            elif city == test_city_name:
                test_img.append(img)
                test_lbl.append(lbl)
            else:
                train_img.append(img)
                train_lbl.append(lbl)
        return train_img, train_lbl, val_img, val_lbl, test_img, test_lbl

    # Let’s instantiate this class and iterate through the data samples. We will print the sizes of first 4 samples and show their landmarks.


#tests
face_dataset =OSMDataset(root_dir='D:/programming/datasets/CITY-OSM/')
face_dataset.findallimages()
for i in range(1):
    sample = face_dataset[i]
    show_sample(sample)
    plt.show()
train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = face_dataset.divideOnTestandTrain('berlin', 'potsdam')


