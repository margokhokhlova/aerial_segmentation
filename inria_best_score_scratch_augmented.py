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
from matplotlib.image import imread


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
from dataloaders.Losses import  make_loss

# In[5]:


import os
import random
import re
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)


# In[6]:


## load data
# define backbone since I will do image processing accordingly
BACKBONE = 'resnet50'
preprocess_input = process_image(BACKBONE)


# In[7]:


# load original dataset
import glob
X_train = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/train_frames/*.png"))
y_train = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/train_masks/*.png"))
X_val = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/val_frames/*.png"))
y_val = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/val_masks/*.png"))
print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


# In[8]:


#load augmented data
ground_truth_images_aug = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/train_frames_copy/train/output/train_original_*.png"))
segmentation_mask_images_aug = sorted(glob.glob("/data/margokat/inria/inria_processed_from_folder512/train_frames_copy/train/output/_groundtruth_(1)_train*.png"))
print('The number of additional training samples is =  %d, validation samples is = %d' % (len(ground_truth_images_aug), len(segmentation_mask_images_aug)))


# In[9]:


# augment the X_train and y_train
X_train = X_train + ground_truth_images_aug 
y_train =y_train + segmentation_mask_images_aug
print('The number of training samples after the data augmentation is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


# In[10]:


# calculate coverage
coverage_train = {}
gr_truth = np.zeros((len(y_train ),1))
for i in range(len(y_train )):
    mask = imread(y_train [i])
    H, W = mask.shape
    gr_truth[i] =  np.sum(mask) / (H*W)
    coverage_train[y_train[i]] = gr_truth[i]

coverage_val = {}
gr_truth = np.zeros((len(y_val ),1))
for i in range(len(y_val )):
    mask = imread(y_val[i])
    H, W = mask.shape
    gr_truth[i] =  np.sum(mask) / (H*W)
    coverage_val[y_val[i]] = gr_truth[i]


# In[15]:


# define parameters for sampling

params = {'dim': (512, 512),
          'batch_size': 10,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': True,
          'Flip': True}

training_generator = InriaDatagenerator(X_train, y_train, Transform = True, stratified_sampling=True, coverage=coverage_train, Process_function = preprocess_input, **params)

params = {'dim': (512, 512),
          'batch_size': 10,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False}

validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, stratified_sampling=True, coverage=coverage_val, Process_function = preprocess_input, **params)
(X,y) = training_generator.__getitem__(500, processing = False)
#for i in range(X.shape[0]):
#    show_sample(X[i,:].astype(int), np.squeeze(y[i,:]))


# In[ ]:


# define model
from dataloaders.custom_metrics import iou_score_batch    
model = Unet(BACKBONE, encoder_weights=None, encoder_freeze = False, activation = 'sigmoid')
Adam_opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(Adam_opt, loss=make_loss('lovasz'), metrics=[iou_score, 'binary_accuracy'])


# In[ ]:


# define callbacks
# train parameters
loss_history = []
weight_path = "/data/margokat/models_saved/inria/{}_weights.best.hdf5".format('resnet50_unet_512_augmented')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto',
                                   epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=6)  # probably needs to be more patient, but kaggle time is limited

tboard = TensorBoard(log_dir='Graph_resnet50_unet_512_aug/', histogram_freq=0,  
          write_graph=True, write_images=True)

callbacks_list = [checkpoint, early, tboard]


# In[ ]:


# fit the model
print('Training the model...')
model.load_weights(weight_path) # if to continue training
loss_history = model.fit_generator(generator=training_generator, steps_per_epoch=np.ceil(len(X_train)/params['batch_size'])*2,
        epochs=100,
        validation_data=validation_generator, validation_steps=int(len(X_val)/params['batch_size']),
        use_multiprocessing=False,
        callbacks=callbacks_list, initial_epoch = 39)


# In[ ]:





# In[ ]:




