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
def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-l', '--load', default=None,type=str, help='name of saved weights to load')
  parser.add_argument('-e', '--epochs', default=10,type=int, help='epochs to train model for')
  parser.add_argument('-s', '--save', default='model',type=str, help='name to save model as')
  parser.add_argument('-d', '--imagedir', default=os.getcwd(),type=str, help='directory where PNG images are stored')
  parser.add_argument('-r', '--real', default='../images/imagenet12/',type=str, help='real image dir')
  parser.add_argument('-a', '--attack', default='../images/adversarials/lbfgs/resnet50/',type=str, help='attack image dir')
  parser.add_argument('-m', '--model_type', default='SVM',type=str, help='Classifier <SVM> <MLP>')
  parser.add_argument('-i', '--input_shape', default=224,type=int, help='first dimension of input images')

  args = parser.parse_args()
  return args


def resnet(outputs):
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = Dropout(.4)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.4)(x)
    predictions = Dense(outputs, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return base_model, model


def load_params():
    x = np.load('./resnet_images.npy')
    y = np.load('./resnet_params.npy')
    return (x, y)


def to_integer(y):
    
    print (y, y.shape)
    y = y[:, 0]
    print y
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


def to_params(y):
    pass


def main(_):
    images, labels = load_params()
    assert len(images) == len(labels)
    print (images.shape, labels.shape)

    images, labels = dropk(images, labels, k=11)
    #labels, decoder = to_integer(labels)
    labels = labels[:, 1]
    print (labels)
    n_labels = int(np.max(labels) + 1)
    labels = np_utils.to_categorical(labels)
    val_idx = int(0.9 * len(images))
    x_train = images[:val_idx]
    x_test = images[val_idx:]
    y_train = labels[:val_idx]
    y_test = labels[val_idx:]

    print (x_train.shape, x_test.shape)
#############################################################
    print ("==> Beginning Session")
    base, model = resnet(n_labels)
    model.fit(x_train, y_train,
              batch_size=32, 
              epochs=5,
              validation_data=(x_test, y_test))

    for i, layer in enumerate(base.layers):
        print (i, layer.name)

    for layer in model.layers[:166]:
        layer.trainable = False
    for layer in model.layers[166:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
   
    model.fit(x_train, y_train,
            batch_size=32, 
            epochs=20,
            validation_data=(x_test, y_test))

    for (x, y) in zip(x_test, y_test):
        l = np.argmax(model.predict(np.expand_dims(x, 0)))
        print ("pred: {} , label: {}".format(l, y[0]))
 

main(0) 
