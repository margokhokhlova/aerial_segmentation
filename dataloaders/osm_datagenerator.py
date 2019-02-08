from __future__ import print_function, division

from skimage import io
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

from dataloaders.dataset_helper import findallimagesosm
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import keras

from keras.applications.vgg16 import preprocess_input

class DataGeneratorOSM(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(512, 512), n_channels_img=3, n_channel_mask=1, shuffle=True, label_mask = 'house'):
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
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
        X = np.empty((self.batch_size, *self.dim, self.n_channels_img))
        y = np.empty((self.batch_size, *self.dim, self.n_channels_lbl))

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

            Xs = np.array(Xp)
            ys = np.array(yp) # convert to numpy
            yp.close()
            Xp.close()

            Xs = preprocess_input(Xs) # VGG processing of an image

            if self.label_mask == 'house':
                ys = np.expand_dims(0.2*(ys[:, :, 2]/255.0) + (1.0 - ys[:, :, 2]/255.0), axis=3)  # take only the red channel of the image and weird deformation to get image between 0.2 and 1

            X[i,] = Xs
            # Store class
            y[i,] = ys

        return X, y

if __name__ == '__main__':
# main file test
    # Parameters
    params = {'dim': (512, 512),
              'batch_size': 32,
              'n_channels_img':3, 'n_channel_mask':1,
              'shuffle': True}


    # Datasets
    partition, labels = findallimagesosm(folder='D:/programming/datasets/OSM_processed_margo/')

    # Generators
    training_generator = DataGeneratorOSM(partition['train'], labels['train'], **params)
    validation_generator = DataGeneratorOSM(partition['validation'], labels['validation'], **params)

    X, y = training_generator
    X1, y1 = validation_generator


