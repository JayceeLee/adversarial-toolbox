import numpy as np
import scipy.misc
import time
import glob

data_dir = '/home/neale/repos/adversarial-toolbox/images/adversarials/mnist_real'

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print "WARNING ONLY FIRST {} MNIST DIGITS".format(limit)
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        images = np.zeros((batch_size, 28*28), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        random_state.shuffle(filelist)
        epoch_count[0] += 1
        for n, i in enumerate(files):

            f = '/im_{}.png'.format(n)
            try:
                image = scipy.misc.imread(path+f)#, mode='RGB')
            except IOError:
                continue
            images[n % batch_size] = image.reshape((784,))
            if n > 0 and n % batch_size == 0:
                images.reshape((1, batch_size, 784))
                yield images

        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    return (
        mnist_generator(train_data, batch_size, n_labelled),
        mnist_generator(dev_data, test_batch_size, n_labelled),
        mnist_generator(test_data, test_batch_size, n_labelled)
    )


def make_generator(path, n_files, batch_size):
    epoch_count = [1]
    filelist = glob.glob(path+'/*.png')

    def get_epoch():
        images = np.zeros((batch_size, 28*28), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        random_state.shuffle(filelist)
        epoch_count[0] += 1
        for n, i in enumerate(files):

            f = '/im_{}.png'.format(n)
            try:
                image = scipy.misc.imread(path+f)#, mode='RGB')
            except IOError:
                continue
            images[n % batch_size] = image.reshape((784,))
            if n > 0 and n % batch_size == 0:
                images.reshape((1, batch_size, 784))
                yield images
    return get_epoch


def load(batch_size, data_dir=data_dir):
    return (
        make_generator(data_dir+'/train', 4999, batch_size),
        make_generator(data_dir+'/val', 499, batch_size),
        make_generator(data_dir+'/test', 499, batch_size)
    )


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0, 0, 0, 0])
        if i == 1000:
            break
        t0 = time.time()
