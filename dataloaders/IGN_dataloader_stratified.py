from __future__ import print_function, division

import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import seaborn as sns
from dataloaders.dataset_helper import findallimagesosm, findallimagesosm_nopartition, load_mask_coverage
import matplotlib.pyplot as plt
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import csv

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

class OSMDataset_stratified(Dataset):
    '''Characterizes an OSM dataset for PyTorch
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel '''
    def __init__(self, list_IDs, labels, label_mask = 'house', transforms=None,  mask_classes =None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.label_mask = label_mask
        self.mask_class = mask_classes


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_name = self.list_IDs[index]
        label_name = self.labels[index]
        # Load data and get label
        try:
            X = io.imread(img_name)
            y = io.imread(label_name)
        except:
            X = None
            y = None
            print(img_name)
            print(label_name)
        if self.label_mask =='house':
            y =np.expand_dims(1.0-(y[:, :, 2]/255.0), axis=3) #take only the red channel of the image


        # if self.transform:
        #     X = self.transform(X) # optional transform

        return (X, y)

    def load_all_files(self):
        self.imgs_df = list()
        self.lbls_df = list()
        self.coverage = list()
        len_imgs = len(self.list_IDs)
        for i in range(len_imgs):
            X,y = self.__getitem__(i)
            #self.imgs_df.append(X)
            #self.lbls_df.append(y)
            self.coverage.append(self.get_coverage(y))
    def get_coverage(self, mask):
        '''return the coverage of the mask per image'''
        H, W, C = mask.shape
        mask = np.squeeze(mask)
        return np.sum(mask) / (H*W)



if __name__ == '__main__':

    train_df = {}

    # Datasets
    X, y = findallimagesosm_nopartition(folder = 'D:/programming/datasets/OSM_processed_margo/')
    print(len(X), len(y))
    masks = load_mask_coverage('D:/programming/datasets/OSM_processed_margo/all.csv')
    mask_values = list(masks.values())
    mask_classes = np.asarray(mask_values, dtype = np.float32)
    for i, s  in enumerate(mask_classes):
        mask_classes[i] = cov_to_class(s)


    train_df["images"] = X
    train_df["labels"] = y
    train_df["coverage"] = np.asarray(mask_values, dtype = np.float32)
    train_df["coverage_class"] = mask_classes

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.distplot(train_df['coverage'], kde=False, ax=axs[0])
    sns.distplot(train_df['coverage_class'], bins=10, kde=False, ax=axs[1])
    plt.suptitle("Buildings coverage")
    axs[0].set_xlabel("Coverage")
    axs[1].set_xlabel("Coverage class")



    # train, val, test split
    X_train, X_val, y_train, y_val,  cov_train, cov_val, cov_cl_train, _ = train_test_split(
        train_df["images"], train_df["labels"], train_df["coverage"], train_df["coverage_class"],
        test_size=0.2, stratify=train_df["coverage_class"], random_state=1234)






    X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(
        X_train, y_train, cov_train,
        test_size=0.2, stratify=cov_cl_train, random_state=12345)


    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.distplot(cov_train, bins=10, kde=False, ax=axs[0])
    sns.distplot(cov_test, bins=10, kde=False, ax=axs[1])
    plt.suptitle("Buildings coverage")
    axs[0].set_xlabel("Train")
    axs[1].set_xlabel("Test")
    print('Done')