#coding: utf-8

import keras
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import argparse

from models.genericnet import generic
from models.vgg6 import vggbn
from models.vgg15 import vgg15
from models import resnet

from generate import data_cifar10

from keras import backend
from keras.utils import np_utils
from PIL import Image
import matplotlib.pyplot as plt

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-m', '--model', default='vgg6', help='model name: vgg6, vgg16, generic')
  parser.add_argument('-p', '--pool', default=0, type=int, help='initial pooling width')
  parser.add_argument('-l', '--load', default=None,type=str, help='name of saved weights to load')
  parser.add_argument('-e', '--epochs', default=10,type=int, help='epochs to train model for')
  parser.add_argument('-d', '--imagedir', default=os.getcwd(),type=str, help='directory where PNG images are stored')

  args = parser.parse_args()
  return args

def load_new_data(args):

    #filelist = glob.glob(args.imagedir+'/*.png')
    filelist = glob.glob(args.imagedir+'/*.jpg')

    x = np.array([np.array(Image.open(fname)) for fname in filelist])

    print "\n==> data shape: ", x.shape

    return x

def pred():


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    keras.backend.set_session(tf.Session(config=config))

    args = load_args()

    x_adv = load_new_data(args)
    _, _, x_cifar, _ = data_cifar10()
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")
    # Create TF session and set as Keras backend session

    # Load model
    print "==> loading model"

    args = load_args()

    if args.model == 'vgg6': model = vggbn(top=False, pool=args.pool)
    if args.model == 'vgg15': model = vgg15(top=False, pool=args.pool)
    if args.model == 'generic': model = generic(top=False, ft=True, pool=args.pool)
    if args.model == 'resnet18': model = resnet.build_resnet_18(args.pool)

    model.load_weights(args.load)

    model.summary()

    model.compile(optimizer='adam',
		  loss='categorical_crossentropy',
		  metrics=['accuracy'])

    res = model.predict_classes(x_adv)

    print res.shape
    # for adv we add up the ones, which are misclassifications
    # for real data we want 1s
    s = sum([1. for i in res if i > 0])


    print res
    print s, "/", len(x_adv)
    #print s, "/", len(x_cifar)

if __name__ == '__main__':

    pred()
