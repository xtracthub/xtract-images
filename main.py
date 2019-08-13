import model
import data
import argparse
import time

image_type_encoding = {"graphics": '1', "map_plots": '2', "maps": '3', "photographs": '4', "scientific_plots": '5'}


# TODO: This should CERTAINLY be two separate functions lol.
def train_or_get_metadata(mode_flag, folder_path, flag, label_file_path, image_path, prediction_file, resize_size, pca_components):
    file_list = None
    if mode_flag == 'predict':
        file_list = [image_path]

    print('get all image data')
    start_time = time.time()
    X, valid_list = data.get_image(file_list, resize_size, data_mode=mode_flag)
    # X, valid_list = "Hey", "young"
    end_time = time.time()
    print('time used to extract image data: ' + str(end_time - start_time))
    # print(X)

    if mode_flag == 'test':
        print('get all label data')
        y = data.get_label_data(label_file_path, image_type_encoding)
        start_time = time.time()
        model.test(X, y, resize_size, pca_components)
        end_time = time.time()
        print('finish')
        print('time used for test: ' + str(end_time - start_time))

    elif mode_flag == 'train':
        print('get all label data')
        y = data.get_label_data(label_file_path, image_type_encoding)

        print("X: {}".format(X))
        print("Y: {}".format(y))

        print('start training')
        start_time = time.time()
        model.train(X, y, resize_size, pca_components)
        end_time = time.time()
        print('finish train, model in clf and pca')
        print('time used for train: ' + str(end_time - start_time))

    elif mode_flag == 'test_predict':
        print('get all label data')
        y = data.get_label_data(label_file_path)
        print('start test predict')
        start_time = time.time()
        model.test_predict(X, y)
        end_time = time.time()
        print('finish test_predict')
        print('time used for test_predict: ' + str(end_time - start_time))

    elif mode_flag == 'predict':
        print('start predict')
        start_time = time.time()
        prediction = model.predict(X)
        end_time = time.time()
        print('finish prediction')
        print('time used to predict: ' + str(end_time - start_time))

        return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image classification: main function')
    parser.add_argument('--mode', type=str, default='test',
                        help='mode to run the program: test, train, predict')
    parser.add_argument('--folder_path', type=str, default='/',
                        help='the path of folder to work on')
    parser.add_argument('--folder_mode', type=int, default=1,
                        help='whether looking through sub-folder, 1 is yes')
    parser.add_argument('--label', type=str, default='',
                        help='if test, input the label file')
    parser.add_argument('--image_path', type=str, default='',
                        help='predict a image')
    parser.add_argument('--prediction_file', type=str, default='prediction.json',
                        help='where to store the prediction')
    parser.add_argument('--resize_to', type=int, default=300,
                        help='size of intermediate image')
    parser.add_argument('--pca_components', type=int, default=30,
                        help='dimension of reduced feature')
    argv = parser.parse_args()

    mode_flag = argv.mode
    folder_path = argv.folder_path
    flag = argv.folder_mode
    label_file_path = argv.label
    image_path = argv.image_path
    prediction_file = argv.prediction_file
    resize_size = argv.resize_to
    pca_components = argv.pca_components

    # TODO: Obvious repeat code.
    if mode_flag == "predict":
        t0 = time.time()
        prediction = train_or_get_metadata(mode_flag, folder_path, flag, label_file_path, image_path, prediction_file,
                                           resize_size, pca_components)

        img_type = next(key for key, value in image_type_encoding.items() if value == str(prediction[0]))
        t1 = time.time()

        meta = {"image-sort": {"img_type": img_type, "total_time": t1-t0}}
        print(meta)

    else:
        train_or_get_metadata(mode_flag, folder_path, flag, label_file_path, image_path, prediction_file, resize_size,
                              pca_components)
