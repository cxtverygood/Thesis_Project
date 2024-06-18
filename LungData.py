
import logging
import numpy as np
from numpy.core.multiarray import ndarray
import scipy.io as sio
from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = 'data/'
FILE_NAME = 'lung.mat'
TEST_SIZE = 1 / 3


def load_lung_data() -> (ndarray, ndarray):
    """
    The lung.mat data set is from: https://jundongl.github.io/scikit-feature/datasets.html

    Dataset characteristics:

    # sample: 203
    # features: 3312
    # output classes: 5

    To expect: high variance

    """
    lung_data = sio.loadmat(DATA_PATH + FILE_NAME)

    X_: ndarray = lung_data['X']
    y_: ndarray = lung_data['Y']

    enc = OneHotEncoder().fit(y_)
    y_ = enc.transform(y_).astype('uint8').toarray()

    logging.info("Available output categories:")
    logging.info(enc.categories_)

    return X_, y_


def train_test_split_normalize(X_: ndarray, y_: ndarray, test_size=TEST_SIZE, random_state=42) \
        -> (ndarray, ndarray, ndarray, ndarray):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=test_size, random_state=random_state)

    normalize = StandardScaler()
    normalize.fit(X_train_)
    X_train_ = normalize.transform(X_train_)
    X_test_ = normalize.transform(X_test_)
    return X_train_, X_test_, y_train_, y_test_
=======
import logging
import scipy.io as sio
from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = 'data/'
FILE_NAME = 'lung.mat'
TEST_SIZE = 1 / 3


def load_lung_data() -> (ndarray, ndarray):
    """
    The lung.mat data set is from: https://jundongl.github.io/scikit-feature/datasets.html

    Dataset characteristics:

    # sample: 203
    # features: 3312
    # output classes: 5

    To expect: high variance

    """
    lung_data = sio.loadmat(DATA_PATH + FILE_NAME)

    X_: ndarray = lung_data['X']
    y_: ndarray = lung_data['Y']

    enc = OneHotEncoder().fit(y_)
    y_ = enc.transform(y_).astype('uint8').toarray()

    logging.info("Available output categories:")
    logging.info(enc.categories_)

    return X_, y_


def train_test_split_normalize(X_: ndarray, y_: ndarray, test_size=TEST_SIZE, random_state=42) \
        -> (ndarray, ndarray, ndarray, ndarray):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=test_size, random_state=random_state)

    normalize = StandardScaler()
    normalize.fit(X_train_)
    X_train_ = normalize.transform(X_train_)
    X_test_ = normalize.transform(X_test_)
    return X_train_, X_test_, y_train_, y_test_
>>>>>>> d783e0f7583ffd178c05528c7021989ff4f9780a
