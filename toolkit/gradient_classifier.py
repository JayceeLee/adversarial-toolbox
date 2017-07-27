#coding: utf-8

import keras
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import argparse

from generate import data_cifar10, gan_cifar10
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

from image_grad import plot_pdf as pltpdf
from scipy.ndimage import filters
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Activation, Dense, Flatten, Dropout
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import color
from scipy import misc

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-l', '--load', default=None,type=str, help='name of saved weights to load')
  parser.add_argument('-e', '--epochs', default=10,type=int, help='epochs to train model for')
  parser.add_argument('-s', '--save', default='model',type=str, help='name to save model as')
  parser.add_argument('-d', '--imagedir', default=os.getcwd(),type=str, help='directory where PNG images are stored')
  parser.add_argument('-a', '--archive', default='fgsm',type=str, help='type of attack image to load')

  args = parser.parse_args()
  return args

def svm(outputs):
    #return LinearSVC(max_iter=10000000,  C=0.6)
    return SVC(C=0.8, cache_size=1000, class_weight=None, coef0=0.0,
               decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
               max_iter=-1, probability=False, random_state=None, shrinking=True,
               tol=0.001, verbose=True)

def mlp(outputs):

    top_model = Sequential()
    top_model.add(Dense(256, activation='relu', input_shape=(1024,)))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))
    return top_model

def load_new_data(args):

    filelist = glob.glob(args.imagedir+'/*.png')

    npy_dir = os.getcwd()+'/npy/'+args.archive+'.npy'

    if not os.path.exists(npy_dir):
        print "==> numpy image archive doesn't exist\nCreating ..."
        x = np.array([np.array(Image.open(fname)) for fname in filelist])
        np.save(npy_dir, x)
    else:
        print "==> loading numpy image archive"
        x = np.load(npy_dir)
    #x = x[4000:7000]
    X_train, _, X_test, _ = data_cifar10()
    #X_train, _, X_test, _ = gan_cifar10()

    # we want to partition dataset equally, according to the smaller set
    # ex. 5000 jsma images + 5000 cifar original images. Instead of 5k+50k
    max_size = min(x.shape[0], X_train.shape[0])
    size_train = int(max_size*0.8)+1
    size_val = int(max_size*0.2)

    # shuffle everything. This is ok because labels are binary

    np.random.shuffle(x)
    np.random.shuffle(X_train)
    np.random.shuffle(X_test)
    print "size of adversarial set: {}".format(x.shape)

    y_adv_train = np.zeros(size_train)
    y_train = np.ones(size_train)
    y_adv_test = np.zeros(size_val)
    y_test = np.ones(size_val)
    x_adv_test  = x[:size_val]
    x_adv_train = x[size_val:]
    x_test  = X_test[:size_val]
    x_train = X_train[:size_train]
    X_train = np.concatenate((x_train, x_adv_train))
    Y_train = np.concatenate((y_train, y_adv_train))
    Y_test  = np.concatenate((y_test, y_adv_test))
    X_test  = np.concatenate((x_test, x_adv_test))
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)
    X_train_grad = np.zeros((len(X_train), 32, 32))
    # Using three channels seems to be too sparse, one channel works
    for i in range(len(X_train)):
        im = X_train[i].astype(np.int32)
        # plot grayscale
        im = color.rgb2gray(im)
        # sobel derivative filters
        #gx, gy = [], []
        #for i in range(3):
        imx = np.zeros(im.shape)
        filters.sobel(im,1,imx)
        imy = np.zeros(im.shape)
        filters.sobel(im,0,imy)
        #    gx.extend(imx)
        #    gy.extend(imy)
        magnitude = np.sqrt(imx**2+imy**2)
        X_train_grad[i] = magnitude

    X_test_grad = np.zeros((len(X_test), 32, 32))
    for i in range(len(X_test)):
        im = X_test[i].astype(np.int32)
        # plot grayscale
        im = color.rgb2gray(im)
        # sobel derivative filters
        #gx, gy = [], []
        #for i in range(3):
        imx = np.zeros(im.shape)
        filters.sobel(im,1,imx)
        imy = np.zeros(im.shape)
        filters.sobel(im,0,imy)
        #    gx.extend(imx)
        #    gy.extend(imy)
        magnitude = np.sqrt(imx**2+imy**2)
        X_test_grad[i] = magnitude

    print "\n==> configured data into\n"
    print "training data shape:   ", X_train_grad.shape
    print "training labels shape: ", Y_train.shape
    print "testing data shape:    ", X_test_grad.shape
    print "testing labels shape   ", Y_test.shape

    return X_train_grad, Y_train, X_test_grad, Y_test, X_test

def plot_pdf(d1, d2):
    print "{} items in d1, \n{} items in d2".format(len(d1), len(d2))
    # Plotting histograms and PDFs
    d1 = np.array(d1).flatten()
    d2 = np.array(d2).flatten()
    from scipy.stats import norm
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data)
    x2 = np.linspace(min(d2), max(d2), len(d2))
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(x2,density(x2))
    plt.show()


    plt.figure()
    x = np.linspace(min(d1), max(d1), len(d1))
    x2 = np.linspace(min(d2), max(d2), len(d2))
    #plt.plot(x, norm.pdf(x, np.mean(d1), np.std(d1)),'r-', lw=3, alpha=0.9, label='norm pdf real')
    plt.plot(x2, norm.pdf(x2, np.mean(d2), np.std(d2)),'b-', lw=3, alpha=0.9, label='norm pdf adversarial')
    plt.suptitle("histogram of gradients on all images")
    #plt.hist(d1, bins=15, normed=True, stacked=True, cumulative=True)
    plt.hist(d2, bins='auto', normed=False, stacked=True, cumulative=True)
    plt.title("mag all samples")
    plt.legend(loc='best', frameon=False)
    plt.show()

def ft():

    args = load_args()
    X_train_grad, Y_train, X_test_grad, Y_test, X_test = load_new_data(args)

    X_train_grad = X_train_grad.reshape(X_train_grad.shape[0], 1024)
    X_test_grad = X_test_grad.reshape(X_test_grad.shape[0], 1024)
    print X_train_grad.shape

    print Y_train
    args = load_args()
    model_type = 'SVM' # change model type <SVM> <MLP>

    #############################################################
    # SVM model training
    if model_type == 'SVM':
        print Y_test
        print "Creating SVM"
        param_grid = [{'C': [.5, .8, 1, 10], 'kernel': ['linear', 'rbf', 'poly'],'gamma': [0.001, 0.0001] }]
        model = svm(2)
        clf = GridSearchCV(model, param_grid)
        print "grid seaching across C, kernels, and gamma"
        clf.fit(X_train_grad, Y_train)
        print sorted(clf.cv_results_.keys())
        correct, fp, fn = 0, 0, 0
        c_grads, n_grads = [], []
        real, ad = [], []
        for i, sample in enumerate(X_test_grad):

            if Y_test[i] == 0.:
                ad.append(sample)
            elif Y_test[i] == 1.:
                real.append(sample)
            pred = clf.predict(sample)[0]
            if pred == Y_test[i]:
                print "correct -- pred: {}\t label: {}".format(pred, Y_test[i])
                correct += 1.
                c_grads.append(sample)
            else:
                print "incorrect -- pred: {}\t label: {}".format(pred, Y_test[i])
                if pred == 0:
                    fn += 1.
                if pred == 1:
                    fp += 1.
                n_grads.append(sample)
                #plt.imshow(X_test[i])
                #plt.show()

        print "\nACC: {}, {}".format(correct / len(X_test_grad), correct)
        print "False Negative: {}, {}".format(fn/len(X_test_grad), fn)
        print "False Positive: {}, {}".format(fp/len(X_test_grad), fp)

        pltpdf(c_grads, n_grads)
        #############################################################

    # MLP model training
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        keras.backend.set_session(tf.Session(config=config))
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
        # Load model
        print "==> loading vgg model"
        model = mlp(2)
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train_grad, Y_train,
                  epochs=args.epochs,
                  batch_size=32,
                  shuffle=True,
                  validation_data=(X_test_grad, Y_test))

        result_dir = os.getcwd()+'/ckpts/detectors/sigmoid_'+args.attack+'/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        model.save_weights(result_dir+args.save)
    ################################################################
if __name__ == '__main__':

    ft()
