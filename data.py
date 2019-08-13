
from __future__ import print_function

from PIL import Image

import numpy as np
import os


w, h = 300, 300


def get_image(file_list, resize_size, data_mode='predict'):
    X = []
    valid_list = []
    idx = 0

    if data_mode == 'predict':
        # TODO: This is a lot of repeat code w/ the train section.
        image = Image.open(file_list[0])  # Because it's only 1 file.
        image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
        image = image.convert('L')
        img_array = list(image.getdata())
        X.append(np.array(img_array))
        valid_list.append(idx)

    else:
        for type_dir in os.listdir("training_data"):

            for filename in os.listdir("training_data/" + type_dir):

                # TODO: Try/Except the image extractor
                #try:
                image = Image.open("training_data/{}/{}".format(type_dir, filename))
                image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
                image = image.convert('L')
                img_array = list(image.getdata())
                X.append(np.array(img_array))
                valid_list.append(idx)
                # except:
                #     pass
                idx += 1
        print("IDX: {}".format(idx))
        print("X1: {}".format(X))
    return X, valid_list


def get_label_data(file_name, image_type_encoding):
    # f = open(file_name)
    # content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip()[0] for x in content]

    res = []
    for type_dir in os.listdir("training_data"):

        for filename in os.listdir("training_data/" + type_dir):
            res.append(int(image_type_encoding[type_dir]))



    return res
