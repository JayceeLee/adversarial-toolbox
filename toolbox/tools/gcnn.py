import sys
sys.path.append('../')
import numpy as np
import keras.backend as K
import plot_gradients as pg
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import Input
from sklearn.decomposition import PCA

from sklearn.utils import shuffle
from sklearn import metrics
from scipy.signal import convolve2d
from scipy import ndimage
from skimage import color
from keras.applications.resnet50 import preprocess_input, decode_predictions
from ResNet50 import ResNet50
from keras.utils import conv_utils

import tensorflow as tf

from scipy import signal

weights_top = '../models/weights/base_models/resnet50_full.h5'
weights_notop = '../models/weights/base_models/resnet50_notop.h5'


class GradientConv2D(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 rank=2,
                 **kwargs):

        super(GradientConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.sx = np.array([
                            [-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]
                           ])

        # self.sx = K.constant(sx, shape=(3, 3))

        self.sy = np.array([
                            [1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]
                           ])

        # self.sy = K.constant(sy, shape=(3, 3))


    def init_sobel_x(self, shape, dtype=None):

        return self.sx.reshape((3, 3, 1, 1))

    def init_sobel_y(self, shape, dtype=None):

        return self.sy.reshape((3, 3, 1, 1))

    def build(self, input_shape):

        if len(input_shape) < 4:
            raise ValueError('Inputs to `GradientConv2D` should have rank 4. '
                             'Recieved input shape: ', str(input_shape))

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('the channel dimension of the inputs to '
                             ' GradientConv2D '
                             'should be defined. Found `None`.')

        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernelx = self.add_weight(name='sobel_x',
                                       shape=kernel_shape,
                                       initializer=self.init_sobel_x,
                                       trainable=False)

        self.kernely = self.add_weight(name='sobel_y',
                                       shape=kernel_shape,
                                       initializer=self.init_sobel_y,
                                       trainable=False)

        self.output_dim = input_shape
        super(GradientConv2D, self).build(input_shape)

    def call(self, inputs):

        im1 = K.expand_dims(inputs[:, :, :, 0], 3)
        im2 = K.expand_dims(inputs[:, :, :, 1], 3)
        im3 = K.expand_dims(inputs[:, :, :, 2], 3)

        sx = self.sx.reshape((3, 3, 1, 1))
        sy = self.sy.reshape((3, 3, 1, 1))

        XconvR = tf.nn.conv2d(im1, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvR = tf.nn.conv2d(im1, sy, strides=[1, 1, 1, 1], padding='SAME')
        XconvG = tf.nn.conv2d(im2, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvG = tf.nn.conv2d(im2, sy, strides=[1, 1, 1, 1], padding='SAME')
        XconvB = tf.nn.conv2d(im3, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvB = tf.nn.conv2d(im3, sy, strides=[1, 1, 1, 1], padding='SAME')

        mag = [
                XconvR*XconvR + YconvR*YconvR,
                XconvG*XconvG + YconvG*YconvG,
                XconvB*XconvB + YconvB*YconvB
              ]

        output = K.concatenate(mag)
        x = K.concatenate([inputs, output], axis=3)

        return x

    def compute_output_shape(self, input_shape):
        print input_shape
        dim = list(input_shape)
        dim[-1] += 3
        output_shape = tuple(dim)
        print output_shape
        return output_shape


def Grad2D(x):

        sx = np.array([
                        [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]
                      ])

        sy = np.array([
                        [1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]
                      ])

        im1 = K.expand_dims(x[:, :, :, 0], 3)
        im2 = K.expand_dims(x[:, :, :, 1], 3)
        im3 = K.expand_dims(x[:, :, :, 2], 3)

        sx = sx.reshape((3, 3, 1, 1))
        sy = sy.reshape((3, 3, 1, 1))

        XconvR = tf.nn.conv2d(im1, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvR = tf.nn.conv2d(im1, sy, strides=[1, 1, 1, 1], padding='SAME')
        XconvG = tf.nn.conv2d(im2, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvG = tf.nn.conv2d(im2, sy, strides=[1, 1, 1, 1], padding='SAME')
        XconvB = tf.nn.conv2d(im3, sx, strides=[1, 1, 1, 1], padding='SAME')
        YconvB = tf.nn.conv2d(im3, sy, strides=[1, 1, 1, 1], padding='SAME')

        mag = [
                XconvR*XconvR + YconvR*YconvR,
                XconvG*XconvG + YconvG*YconvG,
                XconvB*XconvB + YconvB*YconvB
              ]

        output = K.concatenate(mag)

        return output


def Grad2D_output_shape(input_shape):
    return input_shape


def self_aware_loss(y_true, y_pred):

    abstain_penalty = 2.  # ea
    adversarial_penalty = 10.  # eq

    ea = tf.constant(abstain_penalty, shape=(1,))
    eq = tf.constant(adversarial_penalty, shape=(1,))

    label_true = tf.argmax(y_true)
    label_pred = tf.argmax(y_pred)

    xe = tf.nn.softmax(logits=y_pred)

    # we want to compute a whole lottsa shit
    p1 = y_pred[1]
    p11 = K.constant(1) - K.gather(y_pred, label_true[0])
    p12 = tf.multiply(p1, p11)
    p2 = tf.multiply(y_pred[0], eq)
    ai = p12 + p2
    predict_options = K.concatenate([eq, p11])

    def f1(): return K.gather(predict_options, label_pred)
    def f2(): return ea
    L = tf.cond(tf.less(ai[0], ea[0]), f1, f2)
    K.print_tensor(ai)
    K.print_tensor(ea)
    return p12

    return L

def self_aware_decision(y_true, y_true_d, y_pred_d):

    abstain_penalty = 4.  # ea
    adversarial_penalty = 10.  # eq

    ea = abstain_penalty
    eq = adversarial_penalty

    label_true = np.argmax(y_true_d)
    label_resnet = np.argmax(y_true)

    def softmax(x):  # stable
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    sm = softmax(y_pred_d[0])

    p1 = sm[1] * (1 - sm[label_true])
    p2 = sm[0] * eq
    ai = p1 + p2

    # print p1, p2, ai

    if ai < ea: # we should predict
        if label_true == 1:
            res = label_resnet
        else:
            res = .7
    else:
        res = ea

    return res, ai


def collect_gradients(data, arr):

    # Using three channels seems to be too sparse, one channel works
    # But we're going to give it more of and effort with large images
    for i in range(len(data)):
        im = data[i].astype(np.int32)
        gradients = np.zeros(im.shape)
        channels = im.shape[2]
        for j in range(channels):
            imx = np.zeros((224, 224))
            ndimage.sobel(im[:, :, j], 1, imx)
            imy = np.zeros((224, 224))
            ndimage.sobel(im[:, :, j], 0, imy)
            mag = np.sqrt(imx**2+imy**2)
            gradients[:, :, j] = mag

        arr[i] = gradients

    print "\n==> gradient data shape\n", arr.shape

    return arr


def display_predict(model, x, y):

    for im, label in zip(x, y):
        if label[0] == 1:
            z = np.expand_dims(im, axis=0)
            preds = model.predict(z)
            print('Predicted:', decode_predictions(preds, top=3)[0])
            plt.imshow(z[0])
            plt.show()


def extract_features(x_t, arr, layer):

    npoints = 400
    # feed the image to get features at the specified layer
    # we're going to iterate over the feature maps here
    for idx in range(len(x_t)):
        x = x_t[idx]
        features = layer.predict(np.expand_dims(x, 0))
        feature_set = np.zeros((len(features), features.shape[-1]*6))
        fshape = features.shape[-1]
        for m, fmap in enumerate(features):
            K = fmap.shape[-1]
            samples = np.zeros((npoints, K))
            rows = np.random.randint(fmap.shape[0], size=npoints) # we want to draw n pixels
            cols = np.random.randint(fmap.shape[1], size=npoints) # we want to draw n pixels
            for i, (r, c) in enumerate(zip(rows, cols)):
                samples[i] = fmap[r][c]

            means = np.mean(samples, axis=0) # gives K means
            stds = np.std(samples, axis=0)
            pca = PCA()
            samplesT = np.transpose(samples)
            statistics = np.zeros(fshape)
            for i, p in enumerate(samplesT):
                dim = int(np.sqrt(p.shape[0]))
                p = np.reshape(p, (dim, dim))
                pca.fit(p)
                coeffs = pca.transform(p)
                proj = np.transpose(coeffs) * (p - means[i])
                projZ = proj/(stds[i])
                x = 1./npoints * np.linalg.norm(projZ, ord=1)
                statistics[i] = x

            max_samples = np.max(samples, axis=0)
            min_samples = np.min(samples, axis=0)
            samples25 = np.zeros(fshape)
            samples50 = np.zeros(fshape)
            samples75 = np.zeros(fshape)
            for i, p in enumerate(samplesT):
                psort = sorted(p)[::-1]
                samples25[i] = psort[len(p)//4]
                samples50[i] = psort[len(p)//2]
                samples75[i] = psort[3*(len(p)//4)]

            features = np.concatenate((statistics,
                                       max_samples,
                                       min_samples,
                                       samples25,
                                       samples50,
                                       samples75),
                                       axis=0)

            feature_set[m] = features
        arr[idx] = feature_set
    return arr


def load_base_gcnn():

    dim = (224, 224, 3)
    input_tensor = Input(shape=dim)
    G = GradientConv2D(filters=3, kernel_size=3)(input_tensor)
    G = Conv2D(filters=3, kernel_size=3, strides=(1, 1))(G)
    # G = Lambda(Grad2D)(input_tensor)
    # G = Conv2D(filters=3, kernel_size=3, strides=(1, 1))(G)
    return G
    # return input_tensor


def load_clf(model):

    base = model.output
    x = GlobalAveragePooling2D()(base)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_initializer='glorot_normal', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, kernel_initializer='glorot_normal', activation='relu')(x)
    x = Dropout(.5)(x)
    preds = Dense(2, kernel_initializer='glorot_normal', activation='softmax')(x)
    model = Model(model.input, preds)
    return model


def load_resnet(top='vanilla', weight_path=None, gcnn=False, train=True):

    if top == 'vanilla':

        if gcnn is True:
            gcnn = load_base_gcnn()
            model = ResNet50(weights='imagenet', include_top=False, input_tensor=gcnn)
        else:
            model = ResNet50(weights='imagenet', include_top=False)

        if train is True:
	    for layer in model.layers:
             	layer.trainable = False

        model = load_clf(model)

    elif top == 'detector':

        if weight_path is None:

            print "Loading resnet model"
            model = ResNet50(weights='imagenet', include_top=False)

        else:

            if gcnn is True:

                gcnn = load_base_gcnn()
                print "Loading resnet model"
                model = ResNet50(weights=None, include_top=False, input_tensor=gcnn)

            else:

                model = ResNet50(weights=None, include_top=False)

            print "Appending classifier"
            model = load_clf(model)
            model.summary()
            print "Loading weights from {}".format(weight_path)
            model.load_weights(weight_path)

    else:
        raise ValueError('Bad input for top of gcnn, choose vanilla or detector')

    return model


def load_test():

    dim = (224, 224, 3)
    input_tensor = Input(shape=dim)
    # G = Lambda(Grad2D)(input_tensor)
    G = GradientConv2D(filters=3, kernel_size=3)(input_tensor)
    G = Conv2D(3, 3, strides=(1, 1))(G)
    model = Model(input_tensor, G)
    return model


def test_gcnn(train, test, model):

    x_t, y_t = train
    x_v, y_v = test

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print x_t.shape
    for idx, i in enumerate(x_t[:10]):
        print np.argmax(y_t[idx])
        plt.imshow(i)
        plt.show()
        pred = model.predict(np.expand_dims(i, axis=0))
        test = pred[0]
        print test
        print test.shape
        #test = test.reshape((224, 224))
        if test.shape[-1] == 6:
            test = test[:, :, 3:]
        plt.imshow(test)#, cmap='gray')
        plt.show()
    sys.exit(0)


def train_gcnn(train, test, model):

    x_t, y_t = train
    x_v, y_v = test

    # cnn model training
    acc = 0
    conv_acc = 0
    iterations = 0
    while acc < 0.5:
        print "Training iteration ", iterations
        if iterations > 200:
            print "Number of training iterations too great, exiting"
            sys.exit(0)

        # for i, layer in enumerate(model.layers):
        #     print(i, layer.name)
        model.summary()
        print "Compiling classifier with adam"
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print x_t.shape
        print y_t.shape
        print "==> Tuning added FC layers"
        hist = model.fit(x_t,
                         y_t,
                         epochs=8,
                         batch_size=32,
                         shuffle=True,
                         validation_data=(x_v, y_v))

        acc = hist.history['acc'][-1]
    model.save_weights('/tmp/fc_params.h5')
    while conv_acc < 0.7:
        gcnn = load_base_gcnn()
        model = ResNet50(weights=None, include_top=False, input_tensor=gcnn)
        model = load_clf(model)
        model.load_weights('/tmp/fc_params.h5', by_name=True)
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-7:]:
            layer.trainable = True
        for layer in model.layers[:4]:
            layer.trainable = True

        ckpt = ModelCheckpoint('./conv_model.h5', monitor='val_acc', save_best_only=True)
        callbacks = [ckpt]
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        hist = model.fit(x_t,
                         y_t,
                         epochs=30,
                         batch_size=16,
                         shuffle=True,
                         callbacks=callbacks,
                         validation_data=(x_v, y_v))

        conv_acc = hist.history['val_acc'][-1]
        print "accuracy: {}".format(conv_acc)
    return model, (x_v, y_v)


def plot_roc(y_test, y_score, t):

    for q in [0, 1]:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score, pos_label=q)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='magenta',
                lw=lw, label='ROC curve (area = {0:.2f}), t = {1}'.format(roc_auc, t))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()


def train_self_aware_model(x, y, wpath, gt=None):

    # detector should make binary classifications
    detector = load_resnet(top='detector', weight_path=wpath, gcnn=True)
    model = ResNet50(weights='imagenet', include_top=True)
    out_shape = model.layers[-1].output_shape
    print out_shape

    assert len(x) == len(y) == len(gt)
    print "generating predictions through self aware model"
    predicted = 0
    abstained = 0
    statistics = []
    resnet_top5 = []
    for idx, (img, label) in enumerate(zip(x, y)):
        batch_img = np.expand_dims(img, 0)
        detector_proba = detector.predict(batch_img)
        resnet_proba = model.predict(batch_img)  # get adv prediction
        resnet_top = np.argsort(resnet_proba)[0][-5:][::-1]
        pred, prob = self_aware_decision(resnet_proba, label, detector_proba)
        statistics.append([prob, label, gt[idx]])
        resnet_top5.append(resnet_top)

    print "total reals: {}".format(len(np.where(np.argmax(y, axis=-1) == 1)[0]))
    print "total adversarials: {}".format(len(np.where(np.argmax(y, axis=-1) == 0)[0]))
    for t in range(3, 10):
        adv_abstained = 0
        real_abstained = 0
        adv_predicted = 0
        real_predicted = 0
        abstained = 0
        predicted = 0
        penalty = 0
        resnet_accuracy = 0
        roc_preds = np.zeros(len(x))
        roc_labels = np.zeros(len(x))
        print "simulating t = {}...".format(t)
        for idx, (prob, label, truth) in enumerate(statistics):
            label = np.argmax(label)
            roc_labels[idx] = label
            if prob < t:  # we predict
                predicted += 1
                roc_preds[idx] = 1
                if label == 0:
                    adv_predicted += 1
                else:
                    if truth == resnet_top5[idx][0]:  # if gt in top5
                        resnet_accuracy += 1.
                    real_predicted += 1

            elif prob >= 10:  # not good
                penalty += 1

            else:  # we abstained
                roc_preds[idx] = 0
                abstained += 1
                if label == 0:
                    adv_abstained += 1
                else:
                    real_abstained += 1

        assert predicted + abstained + penalty == len(x)
        print "top5 accuracy: {}".format(resnet_accuracy / real_predicted)
        # plot_roc(roc_labels, roc_preds, t)
        print "total abstain: {}".format(abstained)
        print "total predicted: {}".format(predicted)
        print "total errors: {}".format(penalty)
        print "misabstained reals (fn): {}".format(real_abstained)
        print "mispredicted adv (fp): {}".format(adv_predicted)
        print "well classified reals (tp): {}".format(real_predicted)
        print "well abstained adversarials: (tn): {}".format(adv_abstained)
        print "********************************************************"


