import os
import glob

def findallimagesosm(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolders_names = [f.name for f in os.scandir(folder) if f.is_dir()] #find all files in folders
    images_ID = {}
    labels_ID = {}

    for counter, folder in enumerate(subfolders):
        images = glob.glob(folder + "/*img.png")
        labels = glob.glob(folder + "/*lbl.png")
        images_ID[subfolders_names[counter]] = images
        labels_ID[subfolders_names[counter]] = labels

    return images_ID, labels_ID
