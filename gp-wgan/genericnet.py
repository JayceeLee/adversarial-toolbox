
from __future__ import division, print_function, absolute_import
import keras.backend

from keras.models import Model
from keras.layers.core import Activation,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def generic(img_rows=32, img_cols=32, channels=3, top=False, ft=False, pool=0):

    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    if pool > 0:

        model.add(MaxPooling2D(pool_size=(pool,pool), padding='same', input_shape=(3, 32, 32)))
        model.add(Conv2D(48, (3, 3), padding='same'))

    else:
        model.add(Conv2D(48, (3, 3), padding='same', input_shape=(32, 32, 3)))

    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    if pool <=4:
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if top:

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

    if ft:

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

    return model

