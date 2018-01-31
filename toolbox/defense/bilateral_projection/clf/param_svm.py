import os
import sys
import cv2
import keras
import argparse
import tf_models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import color
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras import backend
from sklearn.svm import SVC
from scipy.ndimage import filters
from keras.models import Sequential
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.applications.resnet50 import ResNet50

os.environ['CUDA_VISIBLE_DEVICES']='0'


def load_params():
    x = np.load('./resnet_images.npy')
    y = np.load('./resnet_params.npy')
    return (x, y)


def to_integer(y):
    
    labels = np.unique(y, axis=0)
    n_labels = len(labels)
    y_int = np.zeros(len(y))
    decoder = dict(zip(map(str, range(n_labels)), labels))
    for i, param in enumerate(y):
        for u, unique in enumerate(labels):
            if np.array_equal(param, unique):
                y_int[i] = u

    return y_int, decoder


def dropk(x, y, k):
    unique_y, idx, cnt = np.unique(y, return_index=True, return_counts=True, axis=0)
    del_idx = np.array([])
    for i, c in enumerate(cnt):
        #print ('try {} - {}: {}\n-------------------'.format(i, c, y[idx[i]]))
        if c < k:
            #print ('deleting {} examples with label {}'.format(c, y[idx[i]]))
            label = y[idx[i]]
            for j, data in enumerate(y):
                if np.array_equal(data, label):
                    del_idx = np.append(del_idx, j)
            #print (del_idx.shape)

    y = np.delete(y, del_idx, axis=0)
    x = np.delete(x, del_idx, axis=0)

    print (y.shape)
    return (x, y)


def svm(outputs):
    # return LinearSVC(max_iter=10000000, C=0.1)

    return SVC(C=0.005, cache_size=10000, probability=True, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3,
            gamma='auto', kernel='linear', max_iter=-1,
            random_state=None, shrinking=True,
            tol=0.001)


def collect_gradients(data, dim):

    # Using three channels seems to be too sparse, one channel works

    data_grad = np.zeros((len(data), dim, dim))
    print (data_grad.shape)
    for i in range(len(data)):
        im = data[i].astype(np.int32)
        im = color.rgb2gray(im)
        imx = np.zeros(im.shape)
        filters.sobel(im, 1, imx)
        imy = np.zeros(im.shape)
        filters.sobel(im, 0, imy)
        magnitude = np.sqrt(imx**2+imy**2)
        data_grad[i] = magnitude

    print ("\n==> gradient data shape\n", data_grad.shape)

    return data_grad


def test_svm_generic(clf, data, plot=False):

    correct, fp, fn = 0, 0, 0
    real, adv = [], []
    c_grads, n_grads = [], []
    x_test, y_test = data

    print x_test.shape
    preds = clf.predict(x_test)
    print (preds.shape)
    print (preds)

    for i, pred in enumerate(preds):
        print (pred, y_test[i])
        if pred == y_test[i]:
            print "correct -- pred: {}\t label: {}".format(pred, y_test[i])
            correct += 1.
        else:
            print "incorrect -- pred: {}\t label: {}".format(pred, y_test[i])
    print ("hits: {}%".format(correct/len(y_test)))


def train_basic_svm(train, test):

    d = train[0].shape[1]
    X_train, Y_train = train
    X_test, Y_test = test
    X_train_grad = collect_gradients(X_train, d)
    X_test_grad = collect_gradients(X_test, d)

    dim = d * d
    X_train_grad = X_train_grad.reshape(X_train_grad.shape[0], dim)
    X_test_grad = X_test_grad.reshape(X_test_grad.shape[0], dim)

    print (Y_train)
    
    # SVM model training
    print ("Creating SVM")
    param_grid = [{'C': [1, .005, 10, 100, .5],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.0001]}]

    model = svm(2)
    clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, verbose=10)
    print ("grid seaching across C and kernels {linear, rbf}")
    #Y_train = Y_train[:, 1]
    #Y_test = Y_test[:, 1]
    clf.fit(X_train_grad, Y_train)
    print (sorted(clf.cv_results_.keys()))
    return clf, (X_test_grad, Y_test)


def main(_):
    images, labels = load_params()
    assert len(images) == len(labels)
    print (images.shape, labels.shape)

    images, labels = dropk(images, labels, k=11)
    labels, decoder = to_integer(labels)

    n_labels = int(np.max(labels) + 1)
    # labels = np_utils.to_categorical(labels)
    val_idx = int(0.9 * len(images))
    images, labels = shuffle(images, labels, random_state=0)
    x_train = images[:val_idx]
    x_test = images[val_idx:]
    y_train = labels[:val_idx]
    y_test = labels[val_idx:]
    print (x_train.shape, x_test.shape)
#############################################################
    print ("==> Beginning Session")
    
    clf, test = train_basic_svm((x_train, y_train), (x_test, y_test))
    x_test, y_test = test
    test_svm_generic(clf, (x_test, y_test))
main(0) 
