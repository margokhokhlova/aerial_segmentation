import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.backbones import get_preprocessing as process_image
from segmentation_models.metrics import iou_score

from dataloaders.inria_datagenerator import InriaDatagenerator
from dataloaders.img_helper import  process_for_GAN
from matplotlib.image import imread
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

from dataloaders.img_helper import show_sample, show_sample_gt

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5, decay=0.0000001)


def create_generator(BACKBONE = 'resnet34', input_shape=(None, None, 3), decoder_filters =(512, 256, 128, 64, 32), weights = None ):
    generator = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False, decoder_filters=decoder_filters, activation='sigmoid')
    if weights is not None:
        generator.load_weights(weights)
    generator.compile(loss=bce_jaccard_loss, optimizer=adam_optimizer())
    return generator


# g = create_generator(input_shape =(256,256,3), weights = None)
#
# g.summary()


def create_discriminator(input_shape =(512,512,1)):
    """ the model takes the segmentation mask concatenaed with an original image"""
    discriminator = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=input_shape,
                                       pooling=None, classes=1)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator


# d = create_discriminator()
# d.summary()



# # load data
# BACKBONE = 'resnet34'
# preprocess_input = process_image(BACKBONE)
#

# In[ ]:


# load second dataset
# import glob
# X_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_frames/*.png"))
# y_train = sorted(glob.glob("/data/margokat/inria/clean_data/train_masks/*.png"))
# X_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_frames/*.png"))
# y_val = sorted(glob.glob("/data/margokat/inria/clean_data/val_masks/*.png"))
# print('The number of training samples is =  %d, validation samples is = %d' % (len(X_train), len(X_val)))
#
# coverage_train = {}
# gr_truth = np.zeros((len(y_train ),1))
# for i in range(len(y_train )):
#     mask = imread(y_train [i])
#     H, W = mask.shape
#     gr_truth[i] =  np.sum(mask) / (H*W)
#     coverage_train[y_train[i]] = gr_truth[i]
#
# coverage_val = {}
# gr_truth = np.zeros((len(y_val ),1))
# for i in range(len(y_val )):
#     mask = imread(y_val[i])
#     H, W = mask.shape
#     gr_truth[i] =  np.sum(mask) / (H*W)
#     coverage_val[y_val[i]] = gr_truth[i]





# define parameters for sampling

# params = {'dim': (256, 256),
#           'batch_size': 16,
#           'n_channels_img':3,
#           'n_channel_mask':1,
#           'shuffle': True,
#           'Flip': True}
#
# training_generator = InriaDatagenerator(X_train, y_train, Transform = True, stratified_sampling=False, coverage=coverage_train, Process_function = preprocess_input, **params)

# params = {'dim': (256, 256),
#           'batch_size': 16,
#           'n_channels_img':3,
#           'n_channel_mask':1,
#           'shuffle': False}

#validation_generator = InriaDatagenerator(X_val, y_val, Transform = True, stratified_sampling=False, coverage=coverage_val, Process_function = preprocess_input, **params)


def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(512,512,3))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan_parallel = multi_gpu_model(gan, gpus=2)
    gan_parallel.compile(loss='binary_crossentropy', optimizer='adam',  metrics = ['binary_accuracy'])
    return gan_parallel
#gan = create_gan(d,g)
#gan.summary()


def plot_generated_images(training_generator, epoch, generator, dim=(2,2), figsize=(10,10)):
    steps = int(np.floor(len(training_generator.list_IDs) / training_generator.batch_size))
    X, y = training_generator. __getitem__(np.random.randint(steps))
    generated_images = generator.predict([X])
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(np.squeeze(generated_images[i,:]), cmap='bone')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('/home/margokat/projects/segmentation/gan_output/gan_generated_image %d.png' %epoch)
    generator.save("/data/margokat/models_saved/inria/resnext50_GAN_generator.hdf5")


def training(training_generator, validaion_generator, epochs=1, batch_size=4, generator=None, discriminator=None):
    # Loading the data

    batch_count =  int(np.floor(len(training_generator.list_IDs) / batch_size))

    # Creating GAN
    if generator is None and discriminator is None:
        generator = create_generator()
        discriminator = create_discriminator()
        gan = create_gan(discriminator, generator)
    else:
        gan = create_gan(discriminator, generator)

    gan.summary()


    for e in range(1, epochs + 1):
        # if e == 1:
        #     print('pre-training of the Generator for 5 epochs!')
        #     weight_path = "/data/margokat/models_saved/inria/resnet34_gan.hdf5"
        #     checkpoint = ModelCheckpoint(weight_path, monitor='val_binary_accuracy', verbose=1,
        #                                  save_best_only=True, mode='max', save_weights_only=True)
        #     callbacks = [checkpoint]
        #     generator.fit_generator(generator=training_generator, steps_per_epoch=np.floor(len(training_generator.list_IDs) / batch_size),
        #         epochs=5,validation_data=validation_generator, validation_steps=np.floor(len(validation_generator.list_IDs) / batch_size), use_multiprocessing=True,
        #         callbacks=callbacks)
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_count)):
            # get images and GT
            X_train, y_train = training_generator.__getitem__(np.random.randint(batch_count))

            # Generate fake segmnetation masks
            generated_images = generator.predict(X_train)

            # Get a random set of  real images
            X_real, y_real = training_generator.__getitem__(np.random.randint(batch_count))

            # process the real GT so it is not pure zeros and ones
            y_real_normalized = process_for_GAN(y_real)
            # Construct different batches of  real and fake data
            X_d = np.concatenate([y_real_normalized, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X_d, y_dis)

            # Tricking the noised input of the Generator as real data
            X_train, y_train = training_generator.__getitem__(np.random.randint(batch_count))
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(X_train, y_gen)

        if e == 1 or e % 2 == 0:
            plot_generated_images(epoch = e, generator = generator,  dim=(4,4), training_generator = validaion_generator)



#training(training_generator, validation_generator, 400, batch_size = params['batch_size'])