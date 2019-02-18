import numpy as np
import torch
from torch.utils.data import DataLoader
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import csv

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





if __name__ == '__main__':
    # main file test

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Datasets
    partition, labels = findallimagesosm(folder = 'D:/programming/datasets/OSM_processed_margo/')

    # Parameters
    params = {'dim': (128, 128),
              'batch_size': 12,
              'n_channels_img':3,
              'n_channel_mask':1,
              'shuffle': True}

    # Generators

    masks_coverage = {}
    file_name_coverage = 'D:/programming/datasets/OSM_processed_margo/train/train.csv'
    with open(file_name_coverage) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            masks_coverage[row['ID']] = row['val']

    training_generator = DataGeneratorOSM(partition['train'], labels['train'], coverage=masks_coverage,
                                          stratified_sampling = True,  **params)
    validation_generator = DataGeneratorOSM(partition['validation'], labels['validation'], **params)


    (X,y) = training_generator. __getitem__(0)
    show_sample(X[0,:].astype(int), np.squeeze(y[0,:]))

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

    loss_history += [unet_model.fit_generator(generator=training_generator, steps_per_epoch=2,
                                              epochs=2,
                                              validation_data=validation_generator, validation_steps=300,
                                              use_multiprocessing=False,
                                              callbacks=callbacks_list)]


    # test the data
    testing_generator = DataGeneratorOSM(partition['test'], labels['test'], **params)
    out_parms = unet_model.evaluate_generator(testing_generator, steps=10)
    print('\n')
    for k, v in zip(unet_model.metrics_names, out_parms):
        print(k, '%2.2f' % v)
