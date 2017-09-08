import sys
import numpy as np
sys.path.append('../')
import load_data
import matplotlib.pyplot as plt
from cifar_base import cifar_model
from keras.utils import to_categorical
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle


def improve_detector(train, test, weights):

    print "==> Beginning Session"
    # Load model
    X_train, Y_train = train
    X_val, Y_val = test
    acc = 0.
    conv_acc = 0.
    iterations = 0

    while acc < 0.7:
        print "Training iteration ", iterations
        if iterations > 200:
            print "Number of training iterations too great, exiting"
            sys.exit(0)

        model = cifar_model(top=False, path=weights)
        base = model.output

        x = Dense(512, kernel_initializer='he_normal', activation='relu')(base)
        x = Dropout(.5)(x)
        preds = Dense(2, kernel_initializer='he_normal', activation='softmax')(x)

        for layer in model.layers:
            layer.trainable = False

        model = Model(model.input, preds)

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print "==> Tuning added FC layers"
        hist = model.fit(X_train,
                         Y_train,
                         epochs=5,
                         batch_size=32,
                         shuffle=True,
                         validation_data=(X_val, Y_val))

        acc = hist.history['acc'][-1]

    model.save_weights('/tmp/fc_params.h5')
    while conv_acc < 0.7:

        model.load_weights('/tmp/fc_params.h5')
        for layer in model.layers[:7]:
            layer.trainable = False
        for layer in model.layers[7:]:
            layer.trainable = True

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        hist = model.fit(X_train,
                         Y_train,
                         epochs=5,
                         batch_size=32,
                         shuffle=True,
                         validation_data=(X_val, Y_val))

        conv_acc = hist.history['acc'][-1]
        print "accuracy: {}".format(conv_acc)
        iterations += 1

    return model, hist.history['acc']


def test_cifar(model):

    (x_train, y_train), (x_val, y_val) = load_data.load_real()
    pred = model.predict(x_train, verbose=1)
    print "\t--- ", sum([np.argmax(arr) for arr in pred]), " of ", len(pred)
    pred = model.predict(x_val, verbose=1)
    print "\t--- ", sum([np.argmax(arr) for arr in pred]), " of ", len(pred)
