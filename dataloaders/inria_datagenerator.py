from __future__ import print_function, division
from iteround import saferound

import random
from PIL import Image

from dataloaders.dataset_helper import findallimagesosm
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import keras


def get_stratified_sampling(list_IDs, mask_coverage,batch_size):
    '''
    :param list_IDs: list of image indexes
    :param mask_coverage: corresponding mask coverage as a dictionary imageId-mask coverage values
    :param batch_size:
    :return:
    indexes of the batch size stratified
    '''
    n_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0] # 5 bins

    hist, val =  np.histogram(np.fromiter(mask_coverage.values(), dtype=float), bins = n_bins, normed = 1)

    bin1 = [key for key, val in mask_coverage.items() if float(val) <= 0.2]
    bin2 = [key for key, val in mask_coverage.items() if float(val) > 0.2 and float(val) <= 0.4]
    bin3 = [key for key, val in mask_coverage.items() if float(val) > 0.4 and float(val) <= 0.6]
    bin4 = [key for key, val in mask_coverage.items() if float(val) > 0.6 and float(val) <= 0.8]
    bin5 = [key for key, val in mask_coverage.items() if float(val) > 0.8]

    random.shuffle(bin1)
    random.shuffle(bin2)  # shuffle all mini dict
    random.shuffle(bin3)
    random.shuffle(bin4)
    random.shuffle(bin5)

    samples_per_bin = saferound(hist/sum(hist) * batch_size, places=0)

    indexes =[]
    for i in bin1[:int(samples_per_bin[0])]:
        indexes.append(list_IDs.index(i))
    for i in bin2[:int(samples_per_bin[1])]:
        indexes.append(list_IDs.index(i))
    for i in  bin3[:int(samples_per_bin[2])]:
        indexes.append(list_IDs.index(i))
    for i in bin4[:int(samples_per_bin[3])]:
        indexes.append(list_IDs.index(i))
    for i in bin5[:int(samples_per_bin[4])]:
        indexes.append(list_IDs.index(i))

    return indexes

class DataGeneratorOSM(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(512, 512), n_channels_img=3, n_channel_mask=1, shuffle=True, label_mask = 'house', coverage = None, stratified_sampling = False, Transform = False, Process_function = None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels_img = n_channels_img
        self.n_channels_lbl = n_channel_mask
        self.shuffle = shuffle
        self.label_mask = label_mask
        self.on_epoch_end()
        self.mask_coverage = coverage
        self.stratified_sampling = stratified_sampling
        self.transform = Transform
        self.preprocess_input = Process_function

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if self.stratified_sampling:
            # TO DO get indexes given a distribution
            list_IDs_temp = get_stratified_sampling(self.list_IDs, self.mask_coverage, self.batch_size)
        else:
            # Find list of IDs
            list_IDs_temp = [k for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_img), dtype = np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels_lbl),dtype = np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_name = self.list_IDs[ID]
            label_name = self.labels[ID]
            # Load data and get label

            try:
                # Relative Path
                yp = Image.open(label_name)
                Xp = Image.open(img_name)
            except IOError:
                print("image file is truncated :", img_name, label_name)
                img_name = self.list_IDs[0]
                label_name = self.labels[0] # if wrong - always take the first one
                Xp = Image.open(img_name)
                yp = Image.open(label_name)
            except:
                print("unexpected error with  the file :", img_name, label_name)
                img_name = self.list_IDs[0]
                label_name = self.labels[0]  # if wrong - always take the first one
                Xp = Image.open(img_name)
                yp = Image.open(label_name)
            Xp = Xp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
            yp = yp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
            Xs = np.array(Xp)
            ys = np.array(yp) # convert to numpy
            yp.close()
            Xp.close()

            if self.transform:
                try:
                    Xs = self.preprocess_input(Xs) # VGG processing of an image
                except:
                    img_name = self.list_IDs[0]
                    label_name = self.labels[0]  # if wrong - always take the first one
                    Xp = Image.open(img_name)
                    yp = Image.open(label_name)
                    Xp = Xp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
                    yp = yp.resize((self.dim[0], self.dim[0]), Image.ANTIALIAS)
                    Xs = np.array(Xp)
                    ys = np.array(yp)  # convert to numpy
                    Xs = self.preprocess_input(Xs)  # VGG processing of an image
                    ys = np.float32(ys)
                    yp.close()
                    Xp.close()


            if self.label_mask == 'house':
                ys = np.expand_dims(1.0-(ys[:, :, 2]/255.0), axis=3)  # take only the red channel of the image


            X[i,] = Xs
            # Store class
            y[i,] = ys

        return X, y

    #
    # def Calculate_coverage(self):
    #     ''' the function calculaters the mask for images'''
    #     mask_coverage =
    #
    #     for i in range(len(self.list_IDs)):
    #         self.labels
    #         self.list_IDs
    #         img_name, label_name = self.files[idx]
    #         image = io.imread(img_name)
    #         labels = io.imread(label_name)
    #         train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

if __name__ == '__main__':
# main file test
    # Parameters
    params = {'dim': (512, 512),
              'batch_size': 32,
              'n_channels_img':3, 'n_channel_mask':1,
              'shuffle': False}


    # Datasets
    partition, labels = findallimagesosm(folder='D:/programming/datasets/OSM_processed_margo/')

    # Generators
    training_generator = DataGeneratorOSM(partition['train'], labels['train'], **params)
    validation_generator = DataGeneratorOSM(partition['validation'], labels['validation'], **params)

    for i in range(10):
        X, y = training_generator.__getitem__(i)
