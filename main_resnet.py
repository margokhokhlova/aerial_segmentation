import numpy as np
import torch
import csv

from dataloaders.osm_datagenerator import DataGeneratorOSM
from dataloaders.dataset_helper import findallimagesosm

from dataloaders.img_helper import show_sample

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing as process_image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


if __name__ == '__main__':
    # main file test

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    BACKBONE = 'resnet34'
    preprocess_input = process_image(BACKBONE)

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

    training_generator = DataGeneratorOSM(partition['train'], labels['train'], Transform = True, Process_function = preprocess_input, **params)
    validation_generator = DataGeneratorOSM(partition['validation'], labels['validation'], **params)


    (X,y) = training_generator. __getitem__(0)
    show_sample(X[0,:].astype(int), np.squeeze(y[0,:]))

    # define model
    model = Unet(BACKBONE, encoder_weights='imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])



    #define callbacks
    # train parameters
    loss_history = []
    weight_path = "modelssaved/{}_weights.best.hdf5".format('resnet_unet')
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=15)  # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat]


    # fit model
    loss_history = []

    loss_history += model.fit_generator(generator=training_generator, steps_per_epoch=100,
        epochs=15,
        validation_data=validation_generator, validation_steps=100,
        use_multiprocessing=False,
        callbacks=callbacks_list)