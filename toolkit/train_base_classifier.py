# -*- coding: utf-8 -*-


# Data loading and preprocessing
import keras
import resnet
from vgg15 import vgg15
from genericnet import generic
import sys
import numpy as np
import tensorflow as tf
import argparse

from vgg6 import vggbn
from keras import backend
from keras.utils import np_utils
from tensorflow.python.platform import flags
from PIL import Image
from keras.datasets import cifar10

from generate import data_cifar10

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-m', '--model', default='vgg6', help='model name: vgg6, vgg16, resnet18')
  parser.add_argument('-p', '--pool', default=0, type=int, help='initial pooling width')
  parser.add_argument('-f', '--weights', default='model', help='name of weights to save')
  parser.add_argument('-l', '--load', default=None,type=str, help='name of saved weights to load')
  parser.add_argument('-e', '--epochs', default='100',type=int, help='epochs to train model for')
  args = parser.parse_args()
  return args

args = load_args()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75
keras.backend.set_session(tf.Session(config=config))

X_train, Y_train, X_test, Y_test = data_cifar10()

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

if args.model == 'vgg6': model = vggbn(top=True, pool=args.pool)
if args.model == 'vgg15': model = vgg15(top=True, pool=args.pool)
if args.model == 'generic': model = generic(top=True, pool=args.pool)
if args.model == 'resnet18': model = resnet.build_resnet_18(args.pool)


model.summary()

if args.load is not None:
    model.load_weights(args.load)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=args.epochs,
          batch_size=128,
          validation_data=(X_test, Y_test))

model.save_weights(args.weights+'.h5')

