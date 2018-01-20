import os
import cv2
import keras
import argparse
import tf_models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import color
from keras import backend
from sklearn.svm import SVC
from scipy.ndimage import filters
from keras.models import Sequential
from sklearn.externals import joblib
from bilateral_tf_mim import load_images
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Activation, Dense, Flatten, Dropout

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.environ['CUDA_VISIBLE_DEVICES']='1'
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


def mlp():

    top_model = Sequential()
    top_model.add(Dense(256, activation='relu', input_dim=299*299*3))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dense(3))
    top_model.compile(optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy'])

    return top_model


def load_params():
    x = np.load('./inception_v3_mim_500_params.npy')
    assert x.shape == (500, 3)
    return x

rimages, images = load_images('mim', 'imagenet', 'inception_v3', n=500)
labels = load_params()

print images.shape
print labels.shape
n_samples = len(images)
data = images.reshape((n_samples, -1))
flat_x = np.zeros((len(images), 299*299*3))

for i in range(len(images)):
    flat_x[i] = images[i].flatten()

x_train = flat_x[:400]
x_test = flat_x[400:]
y_train = labels[:400]
y_test = labels[400:]

#############################################################
seed = 7
print "==> Beginning Session"
# Load model
estimators = [('mlp', KerasRegressor(build_fn=mlp, epochs=50, batch_size=5, verbose=0))]
print "==> loading mlp"
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#mlp.fit(x_train, y_train)
        #shuffle=True,
        #validation_data=(x_test, y_test))
################################################################
