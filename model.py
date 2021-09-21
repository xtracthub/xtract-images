from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pickle
import os

def train(X_train, y_train, pca_components=30):
    """Trains an SVC model based on X_train and y_train.

    Parameters:
    X_train (list): List of numpy arrays representing images.
    y_train (list): List of labels for X_train.
    pca_components (int): Dimensions of reduced features.

    Returns:
    Saves pca and model as .sav files.
    """
    dir = os.path.dirname(__file__)
    n_components = pca_components
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

    X_train_pca = pca.transform(X_train)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    pickle.dump(pca, open(f'{dir}/pca_model.sav', 'wb'))
    pickle.dump(clf, open(f'{dir}/clf_model.sav', 'wb'))


def test(X, y):
    """Predicts on a list of features and returns an accuracy score.

    Parameters:
    X (list): List of numpy arrays from images to predict on.
    y (list): List of labels for X.

    Return:
    Prints out the accuracy, recall, precision score.
    """
    dir = os.path.dirname(__file__)
    try:
        pca = pickle.load(open(f'{dir}/pca_model.sav', 'rb'))
        clf = pickle.load(open(f'{dir}/clf_model.sav', 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError("Cannot find model classifier file!")

    X_test = pca.transform(X)
    y_pred = clf.predict(X_test)

    print("Accuracy score: {}".format(accuracy_score(y, y_pred)))
    print("Recall score: {}".format(recall_score(y, y_pred, average='micro')))
    print("Precision score: {}".format(precision_score(y, y_pred, average='micro')))


def predict(X):
    """Predicts on a single numpy array from an image.

    Parameter:
    X (np.ndarray): A single numpy array representing an image.

    Return:
    y_pred (int): Prediction of X that can be mapped to a type from
    the image_type_encoding dictionary.
    """
    dir = os.path.dirname(__file__)
    try:
        pca = pickle.load(open(f'{dir}/pca_model.sav', 'rb'))
        clf = pickle.load(open(f'{dir}/clf_model.sav', 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError("Cannot find model classifier file!")

    X = X.reshape(1, -1)
    X_pca = pca.transform(X)
    y_pred = clf.predict(X_pca)
    return y_pred.item()
