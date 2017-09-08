import keras
import os
import numpy as np
import tensorflow as tf
import argparse
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from scipy.ndimage import filters
from keras import backend
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from skimage import color

import plot_gradients as pg
from load_data import load_real_adv

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

def collect_gradients(data):

# Using three channels seems to be too sparse, one channel works
    data_grad = np.zeros((len(data), (args.input_shape, args.input_shape)))

    for i in range(len(data)):
        im = data[i].astype(np.int32)
        im = color.rgb2gray(im)
        imx = np.zeros(im.shape)
        filters.sobel(im,1,imx)
        imy = np.zeros(im.shape)
        filters.sobel(im,0,imy)
        magnitude = np.sqrt(imx**2+imy**2)
        data_grad[i] = magnitude

    print "\n==> gradient data shape\n", data_grad.shape

    return data_grad

def ft():

    args = load_args()
    s = args.input_shape
    X_train, Y_train, X_test, Y_test = load_real_adv(args.real, args.attack, (s, s))
    X_train_grad = collect_gradients(X_train)
    X_test_grad = collect_gradients(X_test)


    X_train_grad = X_train_grad.reshape(X_train_grad.shape[0], s*s)
    X_test_grad = X_test_grad.reshape(X_test_grad.shape[0], s*s)

    #############################################################
    # SVM model training
    if args.model_type == 'SVM':
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

        print "\nACC: {}, {}".format(correct / len(X_test_grad), correct)
        print "False Negative: {}, {}".format(fn/len(X_test_grad), fn)
        print "False Positive: {}, {}".format(fp/len(X_test_grad), fp)

        print "==> serializing model"
        #joblib.dump(clf, '../models/svm_0.pkl')

        pg.plot_pdf([c_grads, n_grads], ["positive", "negative"])
        #############################################################

    # MLP model training
    if args.model_type == 'MLP':

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        keras.backend.set_session(tf.Session(config=config))
        # Create TF session and set as Keras backend session
        print "==> Beginning Session"
        # Load model
        print "==> loading mlp"
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
