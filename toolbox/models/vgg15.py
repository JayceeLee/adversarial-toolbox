
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
import keras.initializers

def vgg15(img_rows=32, img_cols=32, channels=3, top=False, pool=0):

    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)
    init = keras.initializers.glorot_normal()
    model.add(MaxPooling2D(pool_size=(pool,pool), padding='same', input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(1,1)))

    if top:
    	model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    	model.add(Dense(10))


    return model

