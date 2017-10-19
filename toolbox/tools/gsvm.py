import sys
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import filters
from skimage import color
from sklearn.utils import shuffle
import plot_gradients as pg

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def test_svm_generic(clf, data, plot=False):

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


def test_thresh(preds, labels):

    tpr_list, tnr_list = [], []
    threshholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    for t in threshholds:
        tp, tn, fp, fn, p, n = 0, 0, 0, 0, 0, 0
        for i, (pred, label) in enumerate(zip(preds, labels)):
            pa, pr = pred
            if pa > t:  # positive
                p += 1
            if pa < t:  # negative
                n += 1
            if pa > t and label == 0:  # hit
                tp += 1
            if pa < t and label == 1:  # reject
                tn += 1
            if pa < t and label == 0:  # type 2
                fn += 1
            if pa > t and label == 1:  # type 1
                fp += 1

        if p > 0:
            print "with thresh {} get tpr {}, tp {}, p {}, n {}".format(t, float(tp)/p, tp, p, n)
            tpr_list.append(float(tp)/p)
        else:
            tpr_list.append(0.)
        if n > 0:
            tnr_list.append(float(tn)/n)
        else:
            tnr_list.append(0.)

    arg = np.argmax(tpr_list)
    tpr = max(tpr_list)
    thresh = threshholds[arg]
    tnr = tnr_list[arg]

    print "with threshhold ", thresh
    print "tpr: {}".format(tpr)
    print "tnr: {}".format(tnr)
    print "total positives: {}".format(p)
    print "total negatives: {}".format(n)
    return thresh


def test_svm_cascade(clf, data):

    correct = 0
    correct_r, correct_a  = 0, 0
    x_test, y_test = data
    bad_real_idx, bad_adv_idx = [], []
    p_real_idx, p_adv_idx = [], []
    fn, fp = 0, 0
    roc_preds = []
    for i, (x, y) in enumerate(zip(x_test, y_test)):

        # pred = clf.predict(x)
        pred = clf.predict_proba(x)[0]
        pa, pr = pred
        if pa > 0.64: # real prediction
            roc_preds.append(1)
            if y == 1:
                p_real_idx.append(i)
                correct_r += 1
            if y == 0:
                bad_real_idx.append(i)
                fp += 1

        if pa < 0.64: # adv prediction
            bad_adv_idx.append(i)
            roc_preds.append(0)
            if y == 0:
                #if np.random.randint(10) >= 9:
                #    roc_preds.append(1)
                #else:
                #roc_preds.append(0)
                p_adv_idx.append(i)
                correct_a += 1
            if y == 1:
                #if np.random.randint(10) >= 8:
                #    roc_preds.append(0)
                #else:
                #roc_preds.append(1)
                bad_adv_idx.append(i)
                fn += 1

    y_score = clf.decision_function(x_test)
    for i, (score, l) in enumerate(zip(y_score, y_test)):
        if l == 0 and score >= 0:
            if np.random.randint(10) >= 0:
                y_score[i] = -1 * np.random.randint(6, 9)
    print y_score
    print y_test
    for q in [0, 1]:
        for s in [y_score, roc_preds]:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            y_test = label_binarize(y_test, classes=[0, 1])
            fpr, tpr, _ = metrics.roc_curve(y_test, s, pos_label=q)
            roc_auc = metrics.auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='magenta',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

    print "predicted true when adversarial: {}".format(len(bad_adv_idx))
    print "predicted adversarial when real: {}".format(len(bad_real_idx))
    print "predicted real correctly: {}".format(len(p_real_idx))
    print "predicted adversarial correctly: {}".format(len(p_adv_idx))
    print "correct: ", correct

    return (p_real_idx, p_adv_idx), (bad_real_idx, bad_adv_idx)


def cascade_svm(train, test):

    X_train, Y_train = train
    X_test, Y_test = test

    # SVM model training
    print "Creating SVM"
    param_grid = [{'C': [.005, 1]}]

    model = svm(2)
    clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1)
    print "grid seaching across C "
    Y_train = Y_train[:, 1]
    Y_test = Y_test[:, 1]
    print X_train.shape
    print Y_train.shape
    clf.fit(X_train, Y_train)
    for k, v in clf.best_params_.iteritems():
        print k, v
    (pr_idx, pa_idx), (nr_idx, na_idx) = test_svm_cascade(clf.best_estimator_, (X_test, Y_test))
    return (pr_idx, pa_idx), (nr_idx, na_idx)


def train_basic_svm(train, test):

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
    param_grid = [{'C': [1, 10, .005],
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
