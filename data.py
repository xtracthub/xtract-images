from __future__ import print_function

from PIL import Image

import numpy as np


w, h = 300, 300


def read_file_list(file_name):
    with open(file_name) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
    return content


def get_image(file_list, resize_size):
    X = []
    # file_list = read_file_list(file_name)
    valid_list = []
    idx = 0
    for i in file_list:
        # print(i)
        try:
            image = Image.open(i)
            image = image.resize((resize_size, resize_size), Image.ANTIALIAS)
            image = image.convert('L')
            img_array = list(image.getdata())
            X.append(np.array(img_array))
            valid_list.append(idx)
        except:
            pass
        idx += 1
    return X, valid_list


def get_label_data(file_name, data_choosed):
    f = open(file_name)
    content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip()[0] for x in content]
    res = []
    for i in range(len(content)):
        if content[i] != '0':
            res.append(int(content[i]))
    # for i in range(len(data_choosed)):
    # 	res.append(int(content[data_choosed[i]]))
    return res
