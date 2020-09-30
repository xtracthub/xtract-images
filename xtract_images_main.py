import model
import data
import argparse
import time
import numpy as np

import cv2



image_type_encoding = {"graphics": '1', "map_plots": '2', "maps": '3', "photographs": '4', "scientific_plots": '5'}


def preprocess_data(mode_flag, image_path, resize_size=300):
    """Resizes and extracts numpy arrays for all images.

    Parameters:
    mode_flag (str): Mode of image sampler ('predict', 'test', 'train', 'test_predict').
    image_path (str): File path of image to predict on if mode_flag is 'predict'.
    resize_size (int): Dimension to resize images to.

    Return:
    X (list): List of numpy arrays extracted from resized images.
    y (list or None): List of image labels for X, returns None if mode_flag is predict.
    """
    file_list = [image_path if mode_flag == 'predict' else None]
    y = data.get_label_data(image_type_encoding) if mode_flag is not 'predict' else None
    X, invalid_list = data.get_image(file_list, resize_size, mode_flag)

    if y is not None:
        for invalid_indice in sorted(invalid_list, reverse=True):
            del y[invalid_indice]

    return X, y


def extract_image(mode_flag, image_path, resize_size=300, pca_components=30):
    """Can train a SVC model on images to classify them or predict images
    using a pre-trained model.

    Parameters:
    mode_flag (str): Mode ('predict', 'train', 'test').
    image_path (str): Filepath for image to predict on if mode_flag is 'predict'.
    resize_size (int): Dimension to resize images to.
    pca_components (int): Dimension of reduced features.

    Returns:
    Returns a prediction if mode_flag is 'predict', saves a model if mode_flag is 'train',
    prints metrics if mode_flag is 'test'
    """
    if mode_flag in ['predict', 'train', 'test']:
        X, y = preprocess_data(mode_flag, image_path, resize_size)
        if mode_flag == 'train':
            print("Training model")
            t0 = time.time()
            model.train(X, y, pca_components)
            print("Train time: {}".format(time.time() - t0))
        elif mode_flag == 'predict':
            t0 = time.time()
            prediction = model.predict(X)
            img_type = next(key for key, value in image_type_encoding.items() if value == str(prediction[0]))
            total_time = time.time() - t0
            meta = {"img_type": img_type, "extract time": total_time}

            # Now we get the average RGB of the image
            myimg = cv2.imread(image_path)
            avg_color_per_row = np.average(myimg, axis=0)
            bgr_color = np.average(avg_color_per_row, axis=0)
            rgb_color = np.flipud(bgr_color)

            meta["colors"] = {"bgr": bgr_color, "rgb": rgb_color}

            return meta
        elif mode_flag == 'test':
            model.test(X, y)
    else:
        print("Invalid mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification: main function')

    parser.add_argument('--mode', type=str, default='test', required=True,
                        help='mode to run the program: test, train, predict')
    parser.add_argument('--image_path', type=str, default=None,
                        help='predict a image')
    parser.add_argument('--prediction_file', type=str, default='prediction.json',
                        help='where to store the prediction')
    parser.add_argument('--resize_to', type=int, default=300,
                        help='size of intermediate image')
    parser.add_argument('--pca_components', type=int, default=30,
                        help='dimension of reduced feature')
    argv = parser.parse_args()

    mode_flag = argv.mode
    image_path = argv.image_path
    resize_size = argv.resize_to
    pca_components = argv.pca_components

    meta = extract_image(mode_flag, image_path, resize_size, pca_components)

    print(meta)

