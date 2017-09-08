import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import filters
from skimage import color
import plot_gradients as pg

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def svm(outputs):
    # return LinearSVC(max_iter=10000000,  C=0.6)
    return SVC(C=0.8, cache_size=10000, class_weight=None, coef0=0.0,
               decision_function_shape=None, degree=3,
               gamma='auto', kernel='linear', max_iter=-1,
               probability=False, random_state=None, shrinking=True,
               tol=0.001, verbose=True)


def collect_gradients(data, dim):

    # Using three channels seems to be too sparse, one channel works

    data_grad = np.zeros((len(data), dim, dim))
    print data_grad.shape
    for i in range(len(data)):
        im = data[i].astype(np.int32)
        im = color.rgb2gray(im)
        imx = np.zeros(im.shape)
        filters.sobel(im, 1, imx)
        imy = np.zeros(im.shape)
        filters.sobel(im, 0, imy)
        magnitude = np.sqrt(imx**2+imy**2)
        data_grad[i] = magnitude

    print "\n==> gradient data shape\n", data_grad.shape

    return data_grad


def train_svm(train, test):

    d = train[0].shape[1]
    X_train, Y_train = train
    X_test, Y_test = test
    X_train_grad = collect_gradients(X_train, d)
    X_test_grad = collect_gradients(X_test, d)

    dim = d * d
    X_train_grad = X_train_grad.reshape(X_train_grad.shape[0], dim)
    X_test_grad = X_test_grad.reshape(X_test_grad.shape[0], dim)

    # SVM model training
    print "Creating SVM"
    param_grid = [{'C': [1, 10],
                   'kernel': ['linear'],
                   'gamma': [0.0001]}]
    model = svm(2)
    clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, verbose=10)
    print "grid seaching across C and kernels {linear, rbf}"
    Y_train = Y_train[:, 1]
    Y_test = Y_test[:, 1]
    clf.fit(X_train_grad, Y_train)
    print sorted(clf.cv_results_.keys())
    return clf, (X_test_grad, Y_test)


def test_svm(clf, data, plot=False):

    correct, fp, fn = 0, 0, 0
    real, adv = [], []
    c_grads, n_grads = [], []
    x_test, y_test = data

    for i, sample in enumerate(x_test):
        if y_test[i] == 0.:
            adv.append(sample)
        elif y_test[i] == 1.:
            real.append(sample)
        pred = clf.predict(sample)[0]
        if pred == y_test[i]:
            # print "correct -- pred: {}\t label: {}".format(pred, y_test[i])
            correct += 1.
            c_grads.append(sample)
        else:
            # print "incorrect -- pred: {}\t label: {}".format(pred, y_test[i])
            if pred == 0:
                fn += 1.
            if pred == 1:
                fp += 1.
            n_grads.append(sample)

    # print "\nACC: {}, {}".format(correct / len(x_test), correct)
    # print "False Negative: {}, {}".format(fn/len(x_test), fn)
    # print "False Positive: {}, {}".format(fp/len(x_test), fp)
    if plot:
        pg.plot_pdf([c_grads, n_grads], ["positive", "negative"])
    return correct/len(x_test)
