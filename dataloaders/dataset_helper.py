import os
import glob
import numpy as np
import csv

def findallimagesosm(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders_names = [f.name for f in os.scandir(folder) if f.is_dir()] #find all files in folders
    images_ID = {}
    labels_ID = {}

    for counter, folder in enumerate(subfolders):
        images = sorted(glob.glob(folder + "/*img.png"))
        labels = sorted(glob.glob(folder + "/*lbl.png"))
        images_ID[subfolders_names[counter]] = images
        labels_ID[subfolders_names[counter]] = labels




    return images_ID, labels_ID


def findallimagesosm_nopartition(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders_names = [f.name for f in os.scandir(folder) if f.is_dir()] #find all files in folders
    images_ID = {}
    labels_ID = {}
    all_images_id = []
    all_labels_id = []

    for counter, folder in enumerate(subfolders):
        images = sorted(glob.glob(folder + "/*img.png"))
        labels = sorted(glob.glob(folder + "/*lbl.png"))
        images_ID[subfolders_names[counter]] = images
        labels_ID[subfolders_names[counter]] = labels
        all_images_id = np.concatenate((all_images_id, images), axis = 0)
        all_labels_id= np.concatenate((all_labels_id, labels), axis = 0)




    return all_images_id, all_labels_id


def load_mask_coverage(file_name_coverage):
    masks_coverage = {}
    with open(file_name_coverage) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # print(row['ID'], row['val'])
            masks_coverage[row['ID']] = row['val']

    return masks_coverage



def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i