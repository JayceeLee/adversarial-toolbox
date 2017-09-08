import numpy as np
from sklearn.utils import shuffle

def adv_gen(x, batch_size):

    def get_epoch():
        np.random.shuffle(x)
        for i in xrange(len(x) / batch_size):
            feed = x[i*batch_size:(i+1)*batch_size]
            feed = np.reshape(feed, (-1, 3072))
            yield (np.copy(feed), np.zeros(batch_size))

    return get_epoch

def adv_load(x, batch_size):

    return (adv_gen(x[:int(x.shape[0]*.8)],batch_size), adv_gen(x[int(x.shape[0]*.2):],batch_size))

def inf_train_gen1():
    while True:
        for images, targets in train_gen1():
            yield (images, targets)

def inf_train_gen2():
    while True:
        for images, targets in train_gen2():
            yield (images, targets)

def inf_train_gen_adv():
    while True:
        for images, targets in train_gen_adv():
            yield (images, targets)


def mix_real_adv(g1, g2):
    data1, labels1 = g1.next()
    data2, labels2 = g2.next()
    # cifar labels returned are class labels, need binary
    labels2 = np.ones(data2.shape[0])
    data = np.concatenate((data1, data2))
    labels = np.concatenate((labels1, labels2))
    data, labels = shuffle(data, labels)


