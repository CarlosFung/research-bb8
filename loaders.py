import tensorflow as tf
import numpy as np
import pickle
from layers import *
import cv2

# Split the image mask into a foreground and background mask
# and prepare foreground and background so that it can be used
# with the softmax loss.
# The result are two colums per image.
def prepare_mask(logit_mask, index):
    """
    :param logit_mask: The mask images with [N, w, h, 1]. Values should be scaled [0,1]
    :return: tensor with three columns [N, image h*w, c (bg + obj_fg) ]
                N - number of test samples
                h*w - image size
                c - number of classes - two for foreground - background separation.
    """

    # Obtain boolean values for the mask and for the background
    Ytr_class_labels = np.equal(logit_mask, 1)
    Ytr_bg_labels = np.not_equal(logit_mask, 1)
    Ytr_other_class = np.equal(logit_mask, 999)  # just a random number, as long as not 0/1 -> guarantee to be all false

    # Convert the boolean values into floats -- so that
    # computations in cross-entropy loss is correct
    bit_mask_class = Ytr_class_labels.astype(float)
    bit_mask_background = Ytr_bg_labels.astype(float)
    bit_mask_other = Ytr_other_class.astype(float)

    # combine the images along axis 3

    number_of_classes = 16

    for i in range(number_of_classes):
        if i == 0:
            combined_mask = bit_mask_background
        elif i == index:
            combined_mask = np.concatenate((combined_mask, bit_mask_class), axis=3)
        else:
            combined_mask = np.concatenate((combined_mask, bit_mask_other), axis=3)

    # flattem them all
    Ytr_flat_labels = combined_mask.reshape([-1, np.product(logit_mask[0].shape), number_of_classes])

    Ytr_flat_pure_mask = bit_mask_class.reshape([-1, np.product(logit_mask[0].shape)])

    return Ytr_flat_labels, Ytr_flat_pure_mask

# Load and prepare the data.
# The function loads data from a pickle file and extracts the labels.
# It expects to find the data the same shape with [N, w, h, channels]
def prepare_data_RGB_6DoF(filename):
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)

    (filepath, fullfilename) = os.path.split(filename)
    (filenum, extension) = os.path.splitext(fullfilename)
    obj_index = int(filenum)

    Xtr = data["Xtr"]  # RGB image data [N, w, h, channels]

    Ytr = data["Xtr_mask"]  # the mask [N, w, h, 1]
    Ytr = Ytr / 255  # Scale every value to either 0 or 1. The values in the value are in a range [0,255]
    Ytr, Ytr_pm = prepare_mask(Ytr, obj_index)  # tensor with three columns [N, image h*w, c], [N, image h*w]

    Ytr_pose = data["Ytr_pose"]  # (num_images, x, y, z, qx, qy, qz, qw)

    Ytr_bb8 = data["Ytr_bb8"][:,0:16]  # (num_images, x1, y1, x2, y2, ..., x8, y8), removing center (cx,cy).

    Xte = data["Xte"]

    Yte = data["Xte_mask"]
    Yte = Yte / 255
    Yte, Yte_pm = prepare_mask(Yte, obj_index)

    Yte_pose = data["Yte_pose"]

    Yte_bb8 = data["Yte_bb8"][:,0:16]

    return [Xtr, Ytr, Ytr_pm, Ytr_pose, Ytr_bb8, Xte, Yte, Yte_pm, Yte_pose, Yte_bb8]