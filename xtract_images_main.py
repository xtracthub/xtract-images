import model
import data
import argparse
import time
import numpy as np
import cv2


image_type_encoding = {"graphics": '1', "map_plots": '2', "maps": '3', "photographs": '4', "scientific_plots": '5'}


def classify_image(image_path, resize_size=300, pca_components=30):
    """
    
    """
    X = data.get_image(image_path, resize_size)
    t0 = time.time()
    prediction = model.predict(X)
    print(prediction)
    img_type = next(key for key, value in image_type_encoding.items() if value == str(prediction[0]))
    total_time = time.time() - t0
    meta = {"img_type": img_type, "extract time": total_time}
    myimg = cv2.imread(image_path)
    try:
        avg_color_per_row = np.average(myimg, axis=0)
        bgr_color = np.average(avg_color_per_row, axis=0)
        rgb_color = np.flipud(bgr_color)
    except:
        bgr_color = 0
        rgb_color = 0
        meta["colors"] = {"bgr": bgr_color, "rgb": rgb_color}
        return meta


def train_model(dir=None, resize_size=300, pca_components=30):
    """

    """
    meta = {}
    X = data.get_training_data(dir=dir, resize_size=resize_size)
    y = data.get_training_label(dir=dir, image_type_encoding=image_type_encoding)
    model.train(X_train=X, y_train=y, pca_components=pca_components)
    return meta


def test_model(dir=None, resize_size=300, image_type_encoding=image_type_encoding):
    """

    """
    X = data.get_testing_data(dir=dir, resize_size=resize_size)
    y = data.get_testing_label(dir=dir, image_type_encoding=image_type_encoding)
    stats = model.test(X, y)
    return stats

def execute_extractor(filepath=None, mode_flag='predict'):
    """
    """
    if mode_flag not in ['predict', 'train', 'test']:
        print('Invalid Mode')
        return None
    if not filepath and mode_flag not in ['train', 'test']:
        print('Filename Invalid with provided mode.')
        return None

    if mode_flag == 'predict':
        t0 = time.time()
        metadata = classify_image(image_path=filepath)
        t1 = time.time()
    elif mode_flag == 'train':
        t0 = time.time()
        metadata = train_model(dir=filepath)
        t1 = time.time()
    elif mode_flag == 'test':
        t0 = time.time()
        metadata = test_model(dir=filepath, image_type_encoding=image_type_encoding)
        t1 = time.time()
    metadata.update({'extract time': t1-t0})
    return metadata


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
