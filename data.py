from PIL import Image
import numpy as np
import os
import sys

def get_image(filename, resize_size=300):
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
    assert type(filename) is str, 'filename must be a string'
    try:
        image = Image.open(filename)
        image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
        image = image.convert('L')
        img_array = list(image.getdata())
        X = np.asarray(img_array)
        return X
    except:
        print("Error: ", sys.exc_info()[0])


def get_images(filename_list, resize_size=300):
    """
    """
    assert type(filename_list) is list, 'filename_list must be a list'
    Xs = []
    for filename in filename_list:
        Xs.append(get_image(filename, resize_size))
    return Xs


def get_training_data(resize_size=300):
    """
    
    """
    training_list = []
    for type_dir in os.listdir("training_data"):
        if type_dir[0] != '.':
            for filename in os.listdir("training_data/" + type_dir):
                training_list.append("training_data/{}/{}".format(type_dir, filename))
    training_data = get_images(training_list, resize_size)
    return training_data

def get_label_data(image_type_encoding):
    """Returns label data for the training_data set.

    Parameter:
    image_type_encoding (dictionary): Dictionary mapping image types to integers.

    Return:
    res (list): List of integers representing the labels for images in the training_data set.
    """
    label_data = []
    for type_dir in os.listdir("training_data"):
        if type_dir[0] != '.':
            for _ in os.listdir("training_data/" + type_dir):
                label_data.append(int(image_type_encoding[type_dir]))
    return label_data
