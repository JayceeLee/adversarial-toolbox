import numpy as np
import matplotlib.pyplot as plt
import os
import urllib
import gzip
import cPickle as pickle
import scipy.misc
from glob import glob

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = np.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(targets)

        if n_labelled is not None:
            np.random.set_state(rng_state)
            np.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]), np.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    """
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
        print
        print train_data[0][1]
        print train_data[0][1].shape
        plt.imshow(train_data[0][1].reshape((28, 28)), cmap='gray')
        plt.show()
    """
    data_dir = '../../images/adversarials/mnist/'
    train_files = glob(data_dir+'train/*.png')
    val_files = glob(data_dir+'val/*.png')
    test_files = glob(data_dir+'test/*.png')

    train_data = np.empty((20000, 784))
    dev_data = np.empty((5000, 784))
    test_data = np.empty((5000, 784))
    for i in range(min(5000, len(train_files))):
        im = scipy.misc.imread(train_files[i])
        im = im/255.
        im = im.reshape((784))
        train_data[i] = im
        # import sys
        # sys.exit(0)
    for j in range(5000):
        im = scipy.misc.imread(val_files[i])/255.
        im = im.reshape((784))
        dev_data[j] = im
    for k in range(5000):
        im = scipy.misc.imread(test_files[i])/255.
        im = im.reshape((784))
        test_data[k] = im
    train_data = (train_data, np.ones(len(train_data)))
    dev_data = (dev_data, np.ones(len(dev_data)))
    test_data = (test_data, np.ones(len(test_data)))

    return (
        mnist_generator(train_data, batch_size, n_labelled),
        mnist_generator(dev_data, test_batch_size, n_labelled),
        mnist_generator(test_data, test_batch_size, n_labelled)
    )
