import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def cutimage(image, size = (512,512), path = None, name_indx = '.png'):
    ''' function cuts high res images to a smaller resolution
    and returns the array of num_images.
    In this implementation, the frames do not overlap '''
    H, W, C = image.shape
    num_images = np.floor(image.shape[0]/size[0]) * np.floor(image.shape[1]/size[1]) #check how many images are there
    cropped_images = np.zeros((int(num_images), size[0], size[1], C)) # array N_images, H, W, ColorChannels
    index = 0
    for height in range(0, H-512, 512):
        for width in range(0, W-512, 512):
            cropped_image = image[height:height+512, width:width+512]
            if path is not None:
                new_path = path + str(index).zfill(2) + name_indx
                io.imsave(new_path, cropped_image)


            cropped_images[0,:,:,:] = cropped_image
            index +=1
    return cropped_images

def show_sample(img, lbl,  undoprocessing = True):
    img = img.astype(np.uint8)

    """Show image with labels"""
    mean = [0.485, 0.456, 0.406]  # from KERAS Processing
    std = [0.229, 0.224, 0.225]
    if undoprocessing:
        img = img.astype(np.float64)
        img[:, :,0] *= std[0]
        img[:, :,1]  *= std[1]
        img[:, :,2]  *= std[2]
        img[:, :,0] += mean[0]
        img[:, :,1] += mean[1]
        img[:, :,2] += mean[2]
        img = (255*img).astype(np.int16)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.suptitle("Image")
    plt.imshow(img)
    fig.add_subplot(1,2,2)
    plt.imshow(lbl, cmap = 'bone')
    plt.suptitle("Mask")
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def show_sample_gt(img, lbl, gt, undoprocessing = False):
    """Show image with labels"""
    mean = [0.485, 0.456, 0.406]  # from KERAS Processing
    std = [0.229, 0.224, 0.225]

    if undoprocessing:
        img = img.astype(np.float64)
        img[:, :,0] /= 1/std[0]
        img[:, :,1] /= 1/std[1]
        img[:, :,2] /= 1/std[2]
        img[:, :,0] += mean[0]
        img[:, :,1] += mean[1]
        img[:, :,2] += mean[2]
        img = img[:,:,::-1]
        img = (255*img).astype(np.int16)
    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(img)
    plt.title("Image")
    fig.add_subplot(1,3,2)
    plt.imshow(lbl, cmap = 'bone')
    plt.title("Generated Label")
    fig.add_subplot(1, 3, 3)
    plt.imshow(gt, cmap = 'bone')
    plt.title("Ground Truth")
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

