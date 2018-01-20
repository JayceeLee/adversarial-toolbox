import re
import glob
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.utils import shuffle
from scipy.misc import imread, imresize
from keras.datasets import cifar10
from keras.applications.resnet50 import preprocess_input

in_dir = '../../images/imagenet12/train'


def load_images_tf(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images_tf(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


def load_symmetric(real, adv, n_images=None, shape=299):
    if n_images is None:
        n_images = len(real) - 1
    paths_real = glob.glob(real+'/*.png')
    paths_real.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_adv = glob.glob(adv+'/*.png')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_real = paths_real[-n_images:]
    paths_adv = paths_adv[-n_images:]
    x_real = np.empty((len(paths_real), shape, shape, 3))
    real = load_dir(paths_real, arr=x_real, start=0, end=len(paths_real))
    x_adv = np.empty((len(paths_adv), shape, shape, 3))
    adv = load_dir(paths_adv, arr=x_adv, start=0, end=len(paths_adv))
    return real, adv


def split_shuffle(x, y, val=0.15):
    n_val = int(val*len(x))
    X, Y = shuffle(x, y)
    X_val = X[:n_val]
    X_train = X[n_val:]
    Y_val = Y[:n_val]
    Y_train = Y[n_val:]
    return (X_train, Y_train), (X_val, Y_val)


def preprocess(x, net='resnet'):

    print "preprocessing {} samples for {}".format(len(x), net)
    if net == 'cifar10':
        return x / 255.

    elif net == 'resnet':
        return preprocess_input(x)


def one_hot(y, classes):
    print "encoding samples as one-hot for {} classes".format(classes)
    return to_categorical(y, classes)


def get_detector_labels(y, split):

    print "creating binary detector labels for {} images".format(len(y))
    y[:split] = np.ones(split)
    y[split:] = np.zeros(len(y)-split)
    return y


def load_dir(paths, arr, start=0, end=0):

    assert arr.ndim == 4
    imshape = (arr.shape[1], arr.shape[2], arr.shape[3])
    for idx, i in enumerate(range(start, end)):
        image = imread(paths[idx], mode='RGB')
        image = imresize(image, imshape)
        a = image.astype(np.uint8)
        arr[i] = a
    print "Loaded {} images".format(len(paths))

    return arr


def load_labels(n_labels, y_path):

    with open(y_path, 'rb') as f:
        labels = f.readlines()
    labels = map(int, labels)[:n_labels]
    return labels


def load_ilsvrc_labeled(n_images, im_dir, y_path, adv=False, fn_sorting=True):

    paths = glob.glob(im_dir+'/*.JPEG')
    # sort imagenet files, glob doesn't load in order
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    paths = sorted(paths, key=numericalSort)
    images = np.empty((n_images, 224, 224, 3))
    images = load_dir(paths[:n_images],
                      images,
                      start=0,
                      end=n_images)
    images = preprocess(images, net='resnet')
    labels = load_labels(n_images, y_path)
    return images, labels


def load_real_resnet(n_train, dir=in_dir):

    paths = glob.glob(dir+'/*.JPEG')
    assert len(paths) > 0
    images = np.empty((n_train, 224, 224, 3))
    images = load_dir(paths[:n_train], images, start=0, end=n_train)
    print "Image array shape: {}".format(images.shape)
    return images


def load_all_resnet(dir_adv=None, dir_real=in_dir):

    # The data, shuffled and split between train and test sets:
    paths = glob.glob(dir_real+'/*.JPEG')
    assert len(paths) > 0
    adv_paths = glob.glob(dir_adv+'/*.png')
    assert len(adv_paths) > 0
    min_files = min(len(adv_paths), len(paths))
    files = min_files * 2
    images = np.empty((files, 224, 224, 3))
    y = np.empty(len(images))
    images = load_dir(paths[:min_files], arr=images, start=0, end=min_files)
    images = load_dir(adv_paths[:min_files],
                      arr=images,
                      start=min_files,
                      end=len(images))
    y = get_detector_labels(y, split=min_files)
    (x_t, y_t), (x_v, y_v) = split_shuffle(images, y)
    y_t = one_hot(y_t, 2)
    y_v = one_hot(y_v, 2)
    x_t = preprocess(x_t, net='resnet')
    x_v = preprocess(x_v, net='resnet')
    print "Loaded {} train samples: {}".format(x_t.shape[0], x_t[0].shape)
    print 'Loaded {} validation samples: {}'.format(x_v.shape[0], x_v[0].shape)
    return (x_t, y_t), (x_v, y_v)


def load_real_cifar(classes):

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_t = one_hot(y_train, classes)
    y_v = one_hot(y_test, classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_t = preprocess(x_train, net='cifar10')
    x_v = preprocess(x_test, net='cifar10')

    return (x_t, y_t), (x_v, y_v)
