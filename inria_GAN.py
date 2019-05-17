from dataloaders.inria_datagenerator import InriaDatagenerator
from keras.optimizers import Adam
import numpy as np
import os
print(os.environ['PATH'])
# seg models
# load some dependences
from segmentation_models.backbones import get_preprocessing as process_image
from segmentation_models import Unet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from segmentation_models.metrics import iou_score, f_score

from models.GAN_mine import create_discriminator, create_gan, training as train_gan

# In[ ]:


import os
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
from dataloaders.Losses import  make_loss
from keras.metrics import  binary_accuracy
from keras.utils import multi_gpu_model

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"; 
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

## load data
# define backbone since I will do image processing accordingly
BACKBONE = 'resnext50'
preprocess_input = process_image(BACKBONE)


# In[ ]:


# load second dataset
import glob
X_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_frames/*.png"))
y_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_masks/*.png"))
X_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_frames/*.png"))
y_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_masks/*.png"))
print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))


params = {'dim': (512, 512),
          'batch_size': 6,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': True,
          'Flip': True,
          'Dilation': None}

training_generator = InriaDatagenerator(X_train, y_train, Transform = True, stratified_sampling=False, coverage=None, Process_function = preprocess_input, **params)

params = {'dim': (512, 512),
          'batch_size': 16,
          'n_channels_img':3,
          'n_channel_mask':1,
          'shuffle': False,
          'Dilation': None}

validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, stratified_sampling=False, coverage=None, Process_function = preprocess_input, **params)


def sig_iou_score(y_true, y_pred):
  return iou_score(y_true,y_pred) #tf.math.sigmoid(

def sigm_binary_accuracy(y_true, y_pred):
  return binary_accuracy(y_true, y_pred)  #tf.math.sigmoid(

loss_function ='jaccard'

# define model
model = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze = False, decoder_filters= (512, 256, 128, 64, 32), activation='sigmoid') # decoder_block_type='transpose'
Adam_opt = Adam(lr=0.00008, beta_1=0.995, beta_2=0.989, epsilon=None, decay=0.000001, amsgrad=True)
g = multi_gpu_model(model, gpus=2)
g.compile(Adam_opt, loss=make_loss(loss_function), metrics=[sig_iou_score,sigm_binary_accuracy])

# define callbacks
weight_path = "/data/margokat/models_saved/inria/resnext50_GAN_restnet34.hdf5"
# checkpoint = ModelCheckpoint(weight_path, monitor='val_sig_iou_score', verbose=1,
#                              save_best_only=True, mode='max', save_weights_only=True)
#
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_sig_iou_score', factor=0.8, patience=2, verbose=1, mode='auto',
#                                    epsilon=0.0001, cooldown=5, min_lr=0.0001)
# early = EarlyStopping(monitor="val_loss",
#                       mode="min",
#                       patience=10)  # probably needs to be more patient, but kaggle time is limited
#
# tboard = TensorBoard(log_dir='/home/margokat/projects/segmentation/logs/gans/resnext50_GAN_restnet34', histogram_freq=0,
#           write_graph=True, write_images=True)
#
# callbacks_list = [checkpoint, early, reduceLROnPlat, tboard]
#
#
# # fit the model
# print('Training the Generator model...')
#
# g.fit_generator(generator=training_generator, steps_per_epoch=np.ceil(len(X_train)/params['batch_size']),
#         epochs=5,
#         validation_data=validation_generator, validation_steps=int(len(X_val)/params['batch_size']),
#         use_multiprocessing=True, callbacks=callbacks_list, initial_epoch = 0)
g.load_weights(weight_path)

d = create_discriminator(input_shape=(512,512,1))
d.summary()

print("training GAN")
train_gan(generator=g, discriminator=d, epochs = 100,  batch_size=6, training_generator= training_generator, validaion_generator=validation_generator)



