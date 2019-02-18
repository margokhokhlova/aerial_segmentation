from keras.applications.vgg16 import VGG16 as PTModel
from collections import defaultdict, OrderedDict
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D

#  a pretrained VGG16 model as the encoder portion of a U-Net and thus can benefit from the features already created in the model and only
# focus on learning the specific decoding features. The strategy was used with LinkNet by one of the top placers in the competition.
# source https://www.kaggle.com/kmader/vgg16-u-net-on-carvana



def buildUnet(t0_img, dm_img):

    '''initialize base and build a U net
    t0_img - initial image RGB as a BATCH!
    dm_img - coresponding mask as a BATCH'''
    base_pretrained_model = PTModel(input_shape = t0_img.shape[1:], include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = True
    base_pretrained_model.summary()
    '''Collect Interesting Layers for Model'''
    # collect layers by size so we can make an encoder from them
    layer_size_dict = defaultdict(list)
    inputs = []
    for lay_idx, c_layer in enumerate(base_pretrained_model.layers):
        if not c_layer.__class__.__name__ == 'InputLayer':
            layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
        else:
            inputs += [c_layer]
    # freeze dict
    layer_size_dict = OrderedDict(layer_size_dict.items())
    for k,v in layer_size_dict.items():
        print(k, [w.__class__.__name__ for w in v])

    # take the last layer of each shape and make it into an output
    pretrained_encoder = Model(inputs=base_pretrained_model.get_input_at(0),
                               outputs=[v[-1].get_output_at(0) for k, v in layer_size_dict.items()])
    pretrained_encoder.trainable = False
    n_outputs = pretrained_encoder.predict([t0_img])
    for c_out, (k, v) in zip(n_outputs, layer_size_dict.items()):
        print(c_out.shape, 'expected', k)

    # Build U-Net
    x_wid, y_wid = t0_img.shape[1:3]
    in_t0 = Input(t0_img.shape[1:], name='T0_Image')
    wrap_encoder = lambda i_layer: {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}

    t0_outputs = wrap_encoder(in_t0)
    lay_dims = sorted(t0_outputs.keys(), key=lambda x: x[0])
    skip_layers = 2
    last_layer = None
    for k in lay_dims[skip_layers:]:
        cur_layer = t0_outputs[k]
        channel_count = cur_layer._keras_shape[-1]
        cur_layer = Conv2D(channel_count // 2, kernel_size=(3, 3), padding='same', activation='linear')(cur_layer)
        cur_layer = BatchNormalization()(cur_layer)  # gotta keep an eye on that internal covariant shift
        cur_layer = Activation('relu')(cur_layer)

        if last_layer is None:
            x = cur_layer
        else:
            last_channel_count = last_layer._keras_shape[-1]
            x = Conv2D(last_channel_count // 2, kernel_size=(3, 3), padding='same')(last_layer)
            x = UpSampling2D((2, 2))(x)
            x = concatenate([cur_layer, x])
        last_layer = x
    final_output = Conv2D(dm_img.shape[-1], kernel_size=(1, 1), padding='same', activation='sigmoid')(last_layer)
    unet_model = Model(inputs=[in_t0],
                       outputs=[final_output])
    unet_model.summary()
    return unet_model