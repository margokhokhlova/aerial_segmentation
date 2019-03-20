#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# my data
from dataloaders.inria_datagenerator import InriaDatagenerator
from dataloaders.dataset_helper import load_scv_file
from dataloaders.img_helper import show_sample, show_sample_gt
from segmentation_models.backbones import get_preprocessing as process_image
from sklearn.model_selection import train_test_split


# In[3]:


import os
print(os.environ['PATH'])


# In[4]:


# seg models
# load some dependences
from segmentation_models.backbones import get_preprocessing as process_image
from segmentation_models import Unet, PSPNet, Linknet, FPN
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from segmentation_models.losses import bce_jaccard_loss,bce_dice_loss 
from segmentation_models.metrics import iou_score, f_score
from ml_tricks.sgdr import SGDRScheduler


# In[5]:

from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard

import argparse

# In[5]:


###### PARSING THE ARGUEMENTS

parser = argparse.ArgumentParser(description='image2caption example')

# Task parameters
parser.add_argument('--gpuID', type=str, default="0",
                    help='GPU 0 or 1')

# # Visdom / tensorboard
# parser.add_argument('--visdom-url', type=str, default=None,
#                     help='visdom url, needs http, e.g. http://localhost (default: None)')
# parser.add_argument('--visdom-port', type=int, default=8097,
#                     help='visdom server port (default: 8097')
# parser.add_argument('--log-interval', type=int, default=1, metavar='N',
#                     help='batch interval for logging (default: 1')
# parser.add_argument('--visdomenv',type=str, default='margotest',
#                     help='environment name for Visdom')
args = parser.parse_args()
# env_name = args.visdomenv
# vis_port = args.visdom_port
# visdom-url = args.visdom-url
# vis_logint = args.log-interval
gpu_ID = args.gpuID

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ID
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

## load data
# define backbone since I will do image processing accordingly
BACKBONE = 'mobilenet'
preprocess_input = process_image(BACKBONE)
dictionary_XY = load_scv_file('/data/margokat/inria/img_pairs.csv')
X = list(dictionary_XY.keys())
y = list(dictionary_XY.values())
# train and validation splits

# train and validation splits
X_train, X_val, y_train, y_val = train_test_split(
    X, y,   test_size=0.2, random_state=1234)
print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


# In[7]:


# # load second dataset
# # dictionary_XY_add = load_scv_file('/data/margokat/OSM/train/img_pairs.csv')
# # X_add = list(dictionary_XY_add.keys())
# # y_add = list(dictionary_XY_add.values())

# # and add it to the training part only!
# X_train = X_train + X_add[:10000]
# y_train = y_train +y_add[:10000]
# print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


# In[8]:


# define parameters for sampling

params = {'dim': (512, 512),
          'batch_size': 15,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': True,
          'Flip': True}

training_generator = InriaDatagenerator(X_train, y_train, Transform = True, label_mask = None,  Process_function = preprocess_input, **params)

params = {'dim': (512, 512),
          'batch_size': 15,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False}

validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, label_mask = None, Process_function = preprocess_input, **params)


# In[9]:


# define model
model = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze = True)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score, 'binary_accuracy'])


# In[10]:


# define callbacks
# train parameters
loss_history = []
weight_path = "/data/margokat/models_saved/inria/{}_weights.best.hdf5".format('mobilenetfreze_unet')
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
tboard1 = TensorBoard(log_dir='Graph_mob_unet/',     histogram_freq=2,
    write_graph=False, write_images=True)
callbacks_list = [checkpoint, early, schedule]


# In[13]:


# fit the model
print('Training the model...')
#model.load_weights(weight_path) # if to continue training
loss_history = model.fit_generator(generator=training_generator, steps_per_epoch=np.ceil(len(X_train)/params['batch_size'])*2,
        epochs=45,
        validation_data=validation_generator, validation_steps=int(len(X_val)/params['batch_size']),
        use_multiprocessing=False,
        callbacks=callbacks_list)


# In[28]:




