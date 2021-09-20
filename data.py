from PIL import Image
import numpy as np
import os
import sys

def get_image(file_list, resize_size=300, data_mode='predict'):
    """Retrieves images in file_list as numpy arrays if data_mode is 'predict',
    otherwise it returns the training_data set as numpy arrays.

    Parameters:
    file_list (None or list): List of image paths if data_mode is 'predict', otherwise
    None.
    resize_size (int): Dimension to resize images to.
    data_mode (str): Mode of image sampler ('predict', 'test', 'train').

    Returns:
    X (list): List of numpy arrays taken from images.
    invalid_list (list): List of indices from file_list or training_data that represent
    images that were not able to be processed.
    """
    X = []
    invalid_list = []
    if not(data_mode == 'predict'):
        file_list = []
        for type_dir in os.listdir("training_data"):
            if type_dir[0] != '.':
                for filename in os.listdir("training_data/" + type_dir):
                    file_list.append("training_data/{}/{}".format(type_dir, filename))

    for idx, file in enumerate(file_list):
        try:
            image = Image.open(file)
            image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
            image = image.convert('L')
            img_array = list(image.getdata())
            X.append(np.array(img_array))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            invalid_list.append(idx)

    return X, invalid_list


def get_label_data(image_type_encoding):
    """Returns label data for the training_data set.

    Parameter:
    image_type_encoding (dictionary): Dictionary mapping image types to integers.

    Return:
    res (list): List of integers representing the labels for images in the training_data set.
    """
    res = []
    for type_dir in os.listdir("training_data"):
        if type_dir[0] != '.':
            for filename in os.listdir("training_data/" + type_dir):
                res.append(int(image_type_encoding[type_dir]))

    return res
