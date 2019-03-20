import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


# my data
from dataloaders.inria_datagenerator import InriaDatagenerator
from dataloaders.dataset_helper import load_scv_file
from dataloaders.img_helper import show_sample, show_sample_gt

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from ml_tricks.sgdr import SGDRScheduler

# seg models
# load some dependences
from segmentation_models.backbones import get_preprocessing as process_image
from segmentation_models import Unet, PSPNet, Linknet, FPN
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score, f_score




# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

# define backbone since I will do image processing accordingly
BACKBONE = 'resnet50'
preprocess_input = process_image(BACKBONE)
dictionary_XY = load_scv_file('/data/khokhlov/datasets/inria_processed/img_pairs.csv')
X = list(dictionary_XY.keys())
y = list(dictionary_XY.values())
# put all in the contrainer to make it easier to read
train_df = {}  # dict to store the data
train_df["images"] = X
train_df["labels"] = y
# train and validation splits
X_train, X_val, y_train, y_val = train_test_split(
    train_df["images"], train_df["labels"],
    test_size=0.2, random_state=1234)
print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))



# define parameters for sampling

params = {'dim': (512, 512),
          'batch_size': 10,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': True,
          'Flip': True}



training_generator = InriaDatagenerator(X_train, y_train, Transform = True, label_mask = None,  Process_function = preprocess_input, **params)

params = {'dim': (512, 512),
          'batch_size': 10,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False}

validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, label_mask = None, Process_function = preprocess_input, **params)

 # define model
model = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze = False)
preprocess_input = process_image(BACKBONE)


model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score, 'binary_accuracy'])

#define callbacks
# train parameters
loss_history = []
weight_path = "modelssaved/inria/{}_weights.best.hdf5".format('resnet50_unet')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=6)  # probably needs to be more patient, but kaggle time is limited
schedule = SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-3,
                                     steps_per_epoch = np.ceil(len(X_train)/params['batch_size']),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
callbacks_list = [checkpoint, early, schedule]

# fit the model
print('Training the model...')
model.load_weights(weight_path) # if to continue training
with tf.device('/gpu:0'):
    history = model.fit_generator(generator=training_generator, steps_per_epoch=np.ceil(len(X_train)/params['batch_size'])*2,
        epochs=26,
        validation_data=validation_generator, validation_steps=int(len(X_val)/params['batch_size']),
        use_multiprocessing=False,
        callbacks=callbacks_list)