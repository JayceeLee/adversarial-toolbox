import numpy as np

import os
import urllib
import gzip
import cPickle as pickle
from sklearn.utils import shuffle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()

    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    y = []
    for filename in filenames:
        data, labels = (unpickle(data_dir + '/' + filename))
        all_data.append(data)
        y.append(labels)

    images = np.concatenate(all_data, axis=0)
    targets = np.concatenate(y, axis=0)
    images, targets = shuffle(images, targets, random_state=0)

    def get_epoch():

        for i in xrange(len(images) / batch_size):
            yield (np.copy(images[i*batch_size:(i+1)*batch_size]), np.copy(targets[i*batch_size:(i+1)*batch_size]))

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )
