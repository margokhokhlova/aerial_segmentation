from __future__ import print_function, division

import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataloaders.dataset_helper import findallimagesosm
import matplotlib.pyplot as plt
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import csv

class OSMDataset(Dataset):
    '''Characterizes an OSM dataset for PyTorch
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel '''
    def __init__(self, list_IDs, labels, label_mask = 'house', transforms=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transforms = transforms
        self.label_mask = label_mask


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
    # # main file test
    #
    # # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    #
    # # Datasets
    # partition, labels = findallimagesosm(folder = 'D:/programming/datasets/OSM_processed_margo/')
    #
    #
    # # Parameters
    # params = {'batch_size': 64,
    #           'shuffle': True}
    # max_epochs = 100
    #
    #
    # # Generators
    # training_set = OSMDataset(partition['train'], labels['train'])
    # training_generator = DataLoader(dataset = training_set, **params)
    #
    #
    # (X, y) = training_set.__getitem__(0)
    #
    #
    #
    #
    #
    # validation_set = OSMDataset(partition['validation'], labels['validation'])
    # validation_generator = DataLoader(dataset = validation_set, **params)
    #
    # training_set.load_all_files()
    # dictionary = dict(zip(training_set.list_IDs, training_set.coverage))
    #
    # with open('D:/programming/datasets/OSM_processed_margo/train/train.csv', 'w') as f:
    #     for key in  dictionary .keys():
    #         f.write("%s,%s\n" % (key,  dictionary [key]))
    #
    #
    # # Loop over epochs
    # for epoch in range(max_epochs):
    #     # Training
    #     for local_batch, local_labels in training_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    #         print(local_batch.shape)
    #         print(local_labels.shape)
    #
    masks_coverage = {}
    file_name_coverage  = 'D:/programming/datasets/OSM_processed_margo/validation/validation.csv'
    with open(file_name_coverage) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['ID'], row['val'])
            masks_coverage[row['ID']] = row['val']

    print(masks_coverage)

