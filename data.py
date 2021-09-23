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


def _get_modeling_data(usage='training', dir=None, resize_size=300):
    """
    """
    if not usage:
        return 'Must include usage: training or testing'
    if not dir:
        dir = f'{os.getcwd()}/{usage}_data/'

    training_list = []
    for type_dir in os.listdir(dir):
        if type_dir[0] != '.':
            for filename in os.listdir(f'{dir}/{type_dir}'):
                training_list.append(f'{dir}/{type_dir}/{filename}')
    training_data = get_images(training_list, resize_size)
    return training_data


def _get_labeling_data(image_type_encoding, dir=None, usage='training'):
    """
    """
    if usage not in ['training', 'testing']:
        return 'Must include usage: training or testing'
    if not dir:
        dir = f'{os.getcwd()}/{usage}_data/'
    label_data = []
    for type_dir in os.listdir(dir):
        if type_dir[0] != '.':
            for _ in os.listdir(f'{dir}/{type_dir}'):
                label_data.append(int(image_type_encoding[type_dir]))
    return label_data


def get_training_data(dir=None, resize_size=300):
    data = _get_modeling_data(usage='training',dir=dir, resize_size=resize_size)
    return data


def get_testing_data(dir=None, resize_size=300):
    data = _get_modeling_data(usage='testing',dir=dir, resize_size=resize_size)
    return data


def get_training_label(image_type_encoding, usage='training', dir=None, ):
    """
    """
    if not image_type_encoding:
        return 'Must include image_type_encoding'
    training_label = _get_labeling_data(image_type_encoding=image_type_encoding, dir=dir, usage=usage)
    return training_label


def get_testing_label(image_type_encoding, usage='testing', dir=None, ):
    """
    """
    if not image_type_encoding:
        return 'Must include image_type_encoding'
    training_label = _get_labeling_data(image_type_encoding=image_type_encoding, dir=dir, usage=usage)
    return training_label
