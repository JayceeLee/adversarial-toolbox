import keras
import foolbox
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from foolbox.criteria import TargetClass, Misclassification
from keras.backend.tensorflow_backend import set_session


def config_tf(gpu=0.9):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu
    set_session(tf.Session(config=config))
    keras.backend.set_learning_phase(0)


def load_fmodel(kmodel):
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))
    return fmodel


# input image dimensions
img_rows, img_cols = 28, 28


def load_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
		     activation='relu',
		     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.load_weights('mnist.h5')
    return model


def generate(model, image, targets):

    label = np.argmax(model.predictions(image))
    print "label: ", label
    if targets == 2 and label == 0:
        print "classified real as adversarial"
        return None
    criterion = Misclassification()
    try:
        attack = foolbox.attacks.GradientAttack(model, criterion)
    except:
        return None
    adversarial = attack(image, label, unpack=False)
    return adversarial


def attack():

    config_tf()
    num_classes = 10
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
    else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print 'x_train shape:', x_train.shape
    print x_train.shape[0], 'train samples'
    print x_test.shape[0], 'test samples'

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = load_model(input_shape)
    fmodel = load_fmodel(model)

    for i, image in enumerate(x_train):
        image = image.reshape((28, 28))
        scipy.misc.imsave('./images/run_real/im_{}.png'.format(i), image)
        continue
        pred1 = np.argmax(model.predict(np.expand_dims(image, 0)))
        print "prediction: ", pred1
        # adversarial = generate(fmodel, image, 10)
        if adversarial.image is None:
            continue
        im = adversarial.image.reshape(28, 28)
        print adversarial.image.shape
        pred2 = np.argmax(model.predict(np.expand_dims(adversarial.image, 0)))
        print "new prediction: ", pred2
        if pred1 == pred2:
            continue
        else:
            print "difference: ", np.abs(pred1-pred2)
        #plt.imshow(im, cmap='gray')
        #plt.show()
        scipy.misc.imsave('./images/run0/im_{}.png'.format(i), im)

if __name__ == '__main__':
    attack()
