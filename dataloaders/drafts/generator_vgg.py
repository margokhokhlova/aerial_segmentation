from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os
import numpy as np

dg_args = dict(featurewise_center = False,
                  samplewise_center = False,
                  rotation_range = 5,
                  width_shift_range = 0.01,
                  height_shift_range = 0.01,
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.1],
                  horizontal_flip = True,
                  vertical_flip = False, # no upside down cars
                  fill_mode = 'nearest',
                   data_format = 'channels_last',
               preprocessing_function = preprocess_input)
IMG_SIZE = (512, 512) # slightly smaller than vgg16 normally expects
default_batch_size = 8
core_idg = ImageDataGenerator(**dg_args)
mask_args = dg_args.copy()
mask_args['preprocessing_function'] = lambda x: x/255.0
mask_idg = ImageDataGenerator(**mask_args)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen