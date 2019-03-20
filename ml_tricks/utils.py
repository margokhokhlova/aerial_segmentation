import numpy as np
import pandas as pd
import os
import glob
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
from skimage.transform import resize
from skimage.exposure import rescale_intensity, equalize_hist
import scipy.ndimage as ndi
import skimage.morphology as skm


def get_nn_image(data, file_id, IM_H, IM_W, IM_C=1):
    fp = data.image_path.loc[data.id == file_id].values[0]
    im = imread(fp)
    if IM_C == 3:
        im = gray2rgb(im)
    im = resize(im, (IM_H, IM_W, IM_C), mode='constant', preserve_range=True)
    if np.max(im) > 1:
        im /= 255.
    im = rescale_im(im)
    return im.astype(np.float32)


def get_nn_mask(data, file_id, IM_H, IM_W, IM_C=1):
    fp = data.mask_path.loc[data.id == file_id].values[0]
    im = imread(fp).astype(np.float32)
    im = resize(im, (IM_H, IM_W, IM_C), mode='constant', preserve_range=True)
    if np.max(im) > 1:
        im /= 2**16.
    im = im > 0.5
    return im


def remove_small(mask):
    max_pts = mask.shape[0]*mask.shape[1]
    bw = mask.copy().astype(bool)
    if bw.sum() == 0 or bw.sum() == max_pts:
        return bw
    
    label, _ = ndi.label(bw)
    label = skm.remove_small_objects(label, 200)
    label = skm.remove_small_holes(label, 200, connectivity=2)
    return label
    

def down_sample(im, IM_H, IM_W):
    im = resize(im, (IM_H, IM_W, 1), mode='constant', preserve_range=True)
    return im


def rescale_im(im):
    if (im.min() == im.max()):
        return im
    else:
        return (im - im.min()) / (im.max() - im.min() + 1e-3)


def contrast_stretch(im):
    if (im.min() == im.max()):
        return im
    v_min, v_max = np.percentile(im, (2, 98))
    with np.errstate(divide='raise'):
        try:
            im_out = rescale_intensity(im, in_range=(v_min, v_max))
        except:
            im_out = im
    
    if np.any(np.isnan(im_out)):
        return im
    else:
        return im_out


# hist eq
def hist_eq(im):
    if (im.min() == im.max()):
        return im
    im_out = equalize_hist(im)
    if np.any(np.isnan(im_out)):
        return im
    else:
        return im_out


# Bin values to 11 classes, used for salt-coverage based stratified sampling
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        