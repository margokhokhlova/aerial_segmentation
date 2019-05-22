import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, Concatenate, Lambda,  Flatten, Activation
from keras.models import Model
from tqdm import tqdm
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from keras.callbacks import TensorBoard
import tensorflow as tf
from dataloaders.img_helper import  process_for_cGAN, process_for_GAN
from keras.utils import multi_gpu_model


def adam_optimizer():
    return Adam(lr=0.0001, beta_1=0.0, decay=0.0000001)


def create_generator(BACKBONE = 'resnet34', input_shape=(None, None, 3), decoder_filters =(512, 256, 128, 64, 32), weights = None ):
    generator = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False, decoder_filters=decoder_filters, activation='sigmoid')
    if weights is not None:
        generator.load_weights(weights)
    generator.compile(loss=bce_jaccard_loss, optimizer=adam_optimizer())
    return generator


def write_log(callback, names, logs, batch_no):
    """ helper function for tensorboard"""
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def create_discriminator(input_shape=(512,512,4), n_classes = 1):
    """ the model takes the segmentation mask concatenaed with an original image"""

    base_model = ResNet50(include_top=True, weights=None,
                       input_tensor=None, input_shape=input_shape, classes = n_classes)

    # Add final layers
    x = base_model.output
    x = Activation('sigmoid')(x)


    # This is the model we will train
    discriminator= Model(input=base_model.input, output=x)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator





def create_cgan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(None,None,3))
    x = generator(gan_input)
    x_con =Concatenate(axis=3)([x, gan_input]) #tf.concat([x, gan_input], axis = 3) <- same, but I need to use a layer
    gan_output = discriminator(inputs=x_con)
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
    plt.savefig('/home/margokat/projects/segmentation/gan_output/cgan_generated_image %d.png' %epoch)
    generator.save("/data/margokat/models_saved/inria/resnet50_cGAN_generator18.hdf5")


def training(training_generator, validaion_generator, epochs=1, batch_size=4, generator=None, discriminator=None):
    # Loading the data

    batch_count =  int(np.floor(len(training_generator.list_IDs) / batch_size))

    # Creating GAN
    if generator is None and discriminator is None:
        generator = create_generator()
        discriminator = create_discriminator()
        gan = create_cgan(discriminator, generator)
    else:
        gan = create_cgan(discriminator, generator)

    gan.summary()

    log_path = './logs/discriminator'
    callback = TensorBoard(log_path)
    callback.set_model(discriminator)
    train_names = ['loss', 'binary_accuracy']
    val_names = ['val_loss', 'val_binary_accuracy']

    for e in range(1, epochs + 1):

        print("Epoch %d" % e)
        for batch_n in tqdm(range(batch_count)):
            # get images and GT
            X_train, y_train = training_generator.__getitem__(np.random.randint(batch_count))

            # Generate fake segmnetation masks
            generated_images = generator.predict(X_train)

            # Get a random set of  real images
            X_real, y_real = training_generator.__getitem__(np.random.randint(batch_count))

            # process the real GT so it is not pure zeros and ones, modify the color images using the GT and generated masks
            # process the real GT so it is not pure zeros and ones
            y_real_normalized = process_for_GAN(y_real)

            y_real_conditioned =np.concatenate((y_real_normalized, X_real),axis = 3)
            generated_images_conditioned = np.concatenate((generated_images, X_train),axis = 3) # add RGB to my image

            # Construct different batches of  real and fake data
            X_d = np.concatenate([y_real_conditioned, generated_images_conditioned])


            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            logs = discriminator.train_on_batch(X_d, y_dis)

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
            write_log(callback, train_names, logs, batch_n)
        if e == 1 or e % 2 == 0:
            plot_generated_images(epoch = e, generator = generator,  dim=(4,4), training_generator = validaion_generator)
            # also, check the validation accuracy
            # batch_count_v = int(np.floor(len(validaion_generator.list_IDs) / batch_size))
            # X_val_real, y_val_real = validaion_generator.__getitem__(np.random.randint(batch_count_v))
            # y_val = process_for_GAN(y_val_real)
            #
            # X_val, y_val = validaion_generator.__getitem__(np.random.randint(batch_count_v))
            # # Generate fake segmnetation masks
            # generated_images_v = generator.predict(X_val)
            #
            # y_real_conditioned = np.concatenate((y_val, X_val_real), axis=3)
            # generated_images_conditioned = np.concatenate((generated_images_v, X_val), axis=3)  # add RGB to my image
            #
            # # Construct different batches of  real and fake data
            # X_d = np.concatenate([y_real_conditioned, generated_images_conditioned])
            #
            # # Labels for generated and real data
            # y_dis = np.zeros(2 * batch_size)
            # y_dis[:batch_size] = 0.9
            #
            # # Pre train discriminator on  fake and real data  before starting the gan.
            # discriminator.trainable = True
            # logs = discriminator.train_on_batch(X_d, y_dis)
            # write_log(callback, val_names, logs, e)




#training(training_generator, validation_generator, 400, batch_size = params['batch_size'])