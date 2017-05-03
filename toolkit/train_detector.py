# coding: utf-8

import keras
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import argparse
import resnet

from vgg6 import vggbn
from vgg15 import vgg15
from keras import backend
from genericnet import generic
from generate import data_cifar10

from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Activation, Dense, Flatten, Dropout
from PIL import Image

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-m', '--model', default='vgg6', help='model name: vgg6, vgg16, generic')
  parser.add_argument('-p', '--pool', default=0, type=int, help='initial pooling width')
  parser.add_argument('-l', '--load', default=None,type=str, help='name of saved weights to load')
  parser.add_argument('-e', '--epochs', default='100',type=int, help='epochs to train model for')
  parser.add_argument('-s', '--save', default='model',type=str, help='name to save model as')
  parser.add_argument('-d', '--imagedir', default=os.getcwd(),type=str, help='directory where PNG images are stored')
  args = parser.parse_args()
  return args

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def clf(outputs, model=False):

    if model == True:
	top_model = Sequential()
	top_model.add(Flatten(input_shape=(32, 32, 3)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(2, activation='softmax'))
    else:
	top_model = [
	        Flatten(),
		Dense(256, activation='relu'),
		Dropout(0.5),
		Dense(outputs, activation='softmax'),
	]

    return top_model

def ft():


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    keras.backend.set_session(tf.Session(config=config))

    args = load_args()
    filelist = glob.glob(args.imagedir+'/*.png')

    npy_dir = os.getcwd()+'/npy/'+args.model+str(args.pool)+'.npy'
    if not os.path.exists(npy_dir):
        print "==> numpy image archive doesn't exist\nCreating ..."
        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        np.save(npy_dir, x)
    else:
        print "==> loading numpy image archive"
        x = np.load(npy_dir)

    print "size of adversarial set: {}".format(x.shape)
    y = np.zeros(40000)
    y_test = np.zeros(10000)
    Y1 = np.ones(50000)
    Y2 = np.ones(10000)
    X, Y, X_test, Y_test = data_cifar10()
    x_test = x[:10000]
    x = x[10000:]

    X_train = np.concatenate((X, x))
    Y_train = np.concatenate((Y1, y))
    Y_test  = np.concatenate((Y2, y_test))
    X_test  = np.concatenate((X_test, x_test))

    Y_train = np_utils.to_categorical(Y_train, 2)
    Y_test = np_utils.to_categorical(Y_test, 2)

    print X_train.shape
    print Y_train.shape
    print X_test.shape
    print Y_test.shape
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")
    # Create TF session and set as Keras backend session

    print "==> Beginning Session"
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 2., 1. - label_smooth)

    # Load model
    print "==> loading vgg model"

    args = load_args()

    if args.model == 'vgg6': model = vggbn(top=False, pool=args.pool)
    if args.model == 'vgg15': model = vgg15(top=False, pool=args.pool)
    if args.model == 'generic': model = generic(top=False, pool=args.pool)
    if args.model == 'resnet18': model = resnet.build_resnet_18(args.pool)

    model.load_weights(args.load, by_name=True)

    classifier = clf(2, model=False)


    model.summary()

    for layer in classifier:
	model.add(layer)
    model.summary()

    model.compile(optimizer='adam',
		  loss='categorical_crossentropy',
		  metrics=['accuracy'])

    model.fit(X_train, Y_train,
	      epochs=10,
	      batch_size=128,
              shuffle=True,
	      validation_data=(X_test, Y_test))

    result_dir = os.getcwd()+'/models/detectors/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model.save_weights(result_dir+args.save)

if __name__ == '__main__':

    ft()
