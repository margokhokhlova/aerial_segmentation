from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import glob
from skimage import io, transform
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


from img_helper import cutimage


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
    """City Landscapes dataset."""

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
        self.batch_end = False
        self.current_batch = 0 # needed for the next batch function

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

    def getimage(self, name_with_path):
        img = io.imread(name_with_path)
        return img

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
        ''' returns testing and training images, depending on training, validation and test sets
         only the path to images is return and not the images themself, since they are way too big'''
        train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = [], [], [], [], [], []
        for i in range(len(self.files)):
            sample = self.files[i]
            city = self.ids[i]
            img, lbl = sample
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

    # # TO DO: get batch of images
    # def getbatchofdata(self, batch_size, data_img, data_labels, size = (512,512)):
    #     ''' loads images, crops them to 512x512 values
    #     attention! so far batch size should be batch_size%20 = 0 '''
    #     if self.current_batch == 0:
    #         data_img, data_labels = shuffle(data_img, data_labels, random_state=0)
    #
    #     #weird thing to check the number of images I am cutting into
    #     test_img = self.getimage(data_img[0])
    #     test_img_array = cutimage(test_img)
    #     N,_,_,_ = test_img_array.shape()
    #
    #     batch_size = batch_size/N # should be %20 = 0
    #     # TO DO: modify it so he batch size can be of any size
    #
    #     img_batch = data_img[self.current_batch*batch_size:self.current_batch*batch_size*2]
    #     lbl_batch = data_labels[self.current_batch*batch_size:self.current_batch*batch_size*2]




#tests
face_dataset =OSMDataset(root_dir='D:/programming/datasets/CITY-OSM/')
face_dataset.findallimages()
for i in range(1):
    sample = face_dataset[i]
    show_sample(sample)
    plt.show()
train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = face_dataset.divideOnTestandTrain('berlin', 'potsdam')


# pipeline to modify the images so they are 512x512
path ="D:/programming/datasets/OSM_processed_margo/validation/"
for i in range(len(val_img)):
    img = face_dataset.getimage(val_img[i])
    lbl = face_dataset.getimage(val_lbl[i])
    name = path + str(i).zfill(6)
    name_indx = "_img.png"
    _ = cutimage(img, size=(512, 512), path=name, name_indx=name_indx)
    name_indx = "_lbl.png"
    _ = cutimage(lbl, size=(512, 512), path=name, name_indx=name_indx)





