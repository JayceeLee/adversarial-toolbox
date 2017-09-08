import numpy as np
import glob
from keras.utils import to_categorical
from sklearn.utils import shuffle
from scipy.misc import imread, imresize
from keras.datasets import cifar10


def load_real():

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def load_dir(f, suff='.png'):

    paths = glob.glob(f + '*' + suff)
    images = np.empty((len(paths), 32, 32, 3))
    for i, path in enumerate(paths):
        images[i] = imread(path)

    return images


def load_real_adv(dir_adv, shape, suff='.png'):
    # read in and level datasets
    adversarial_paths = glob.glob(dir_adv+'*'+suff)
    adversarials = []
    (x_train_real, y_train_real), (x_val_real, y_val_real) = load_real()
    adversarials = np.empty((len(adversarial_paths), shape[0], shape[1], 3))
    print adversarials.shape
    print len(adversarial_paths)
    for i, path in enumerate(adversarial_paths):
        adversarials[i] = imread(path)
    l_train = len(adversarials)
    l_val = int(l_train * 0.2)
    np.random.shuffle(x_train_real)
    np.random.shuffle(x_val_real)
    np.random.shuffle(adversarials)
    # create validation and binary label sets
    real_train = np.array(x_train_real[:l_train-l_val])
    real_val = np.array(x_val_real[:l_val])
    adversarial_train = np.array(adversarials[l_val:])
    adversarial_val = np.array(adversarials[:l_val])
    print real_train.shape, adversarial_train.shape
    print real_val.shape, adversarial_val.shape
    y_real_train = np.ones(len(real_train))
    y_real_val = np.ones(len(real_val))
    y_adversarial_train = np.zeros(len(adversarial_train))
    y_adversarial_val = np.zeros(len(adversarial_val))
    assert len(adversarial_train) == len(real_train)
    assert len(y_real_train) == len(y_adversarial_train)
    # push datasets together and shuffle deterministically

    X_train = np.concatenate((np.array(real_train), np.array(adversarial_train)))
    Y_train = np.concatenate((np.array(y_real_train), np.array(y_adversarial_train)))
    Y_val = np.concatenate((np.array(y_real_val), np.array(y_adversarial_val)))
    X_val = np.concatenate((np.array(real_val), np.array(adversarial_val)))
    X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)
    Y_train = to_categorical(Y_train, 2)
    Y_val = to_categorical(Y_val, 2)
    # print statistics
    print "\n==> configured data into\n"
    print "training data shape: {}, {}".format(X_train.shape, Y_train.shape)
    print "testing data shape: {}l, {}".format(X_val.shape, Y_val.shape)

    return (X_train, Y_train), (X_val, Y_val)
