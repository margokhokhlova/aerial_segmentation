import numpy as np
import torch
from torch.utils.data import DataLoader
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import os
import tensorflow as tf

from dataloaders.osm_dataloader import OSMDataset
from dataloaders.osm_datagenerator import DataGeneratorOSM
from dataloaders.dataset_helper import findallimagesosm

from models.Unet_keras import buildUnet
from dataloaders.img_helper import show_sample


#functions for the loss
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.0*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)



# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

if __name__ == '__main__':
    # main file test

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Datasets
    partition, labels = findallimagesosm(folder = '/data/khokhlov/datasets/OSM_processed_margo/')

    # Parameters
    params = {'dim': (512, 512),
              'batch_size': 12,
              'n_channels_img':3,
              'n_channel_mask':1,
              'shuffle': True}

    # Generators


    training_generator = DataGeneratorOSM(partition['train'], labels['train'], **params)
    validation_generator = DataGeneratorOSM(partition['validation'], labels['validation'], **params)


    (X,y) = training_generator. __getitem__(0)
    #show_sample(X[0,:].astype(int), np.squeeze(y[0,:]))

    unet_model = buildUnet(X, y)
    unet_model.compile(optimizer=Adam(1e-3, decay=1e-6),
                       loss=dice_p_bce,
                       metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

    # train parameters
    loss_history = []
    weight_path = "modelssaved/{}_weights.best.hdf5".format('vgg_unet')
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=15)  # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    loss_history += [unet_model.fit_generator(generator=training_generator, steps_per_epoch=300,
                                              epochs=50,
                                              validation_data=validation_generator, validation_steps=300,
                                              use_multiprocessing=False,
                                              callbacks=callbacks_list)]
