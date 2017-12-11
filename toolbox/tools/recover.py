# import sys
import cv2
import gcnn
import load_data
import numpy as np
# import matplotlib.pyplot as plt

from glob import glob
from ResNet50 import ResNet50
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import preprocess_input

adv_weights = '../models/weights/detectors/lbfgs/imagenet/iter_0.h5'
adv_dir = '../../images/adversarials/lbfgs/imagenet/symmetric/resnet_test0'
real_images = '../../images/imagenet12/recovery_set_all'


def predictions(model, x, proba=False, topk=1):

    assert len(x) > 0

    if proba:
        iset = np.empty((len(x), 224, 224, 3))
        for i, im in enumerate(x):
            im = np.expand_dims(im, 0)
            proba = model.predict(im)
            iset[i] = proba
    else:
        iset = []
        for im in x:
            im = np.expand_dims(im, 0)
            if topk > 1:
                preds = model.predict(im)
                labels = np.argsort(-preds)[0][:topk]
                iset.append(labels)
            else:
                label = np.argmax(model.predict(im))
                iset.append(label)

    return iset


def load_vanilla_model():

    return ResNet50(weights='imagenet', include_top=True)


def load_adv_gcnn(weights):

    base = gcnn.load_base_gcnn()
    model = ResNet50(weights=None, include_top=False, input_tensor=base)
    model = gcnn.load_clf(model)
    model.load_weights(weights)

    return model


def bilateral(arr, f, sc, ss):

    assert len(arr) > 0 and arr.ndim == 4

    arr = arr.astype(np.float32)
    filtered = np.empty(arr.shape)
    for i, image in enumerate(arr):
        filtered[i] = cv2.bilateralFilter(image, f, sc, ss)
    return filtered


def load_recovery_model(bilateral=False):

    dim = (224, 224, 3)
    inputs = Input(shape=dim)
    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same')(inputs)
    if bilateral is True:
        avg_model = Model(inputs=inputs, outputs=x)
        model = ResNet50(weights='imagenet', include_top=True)
        return (avg_model, model)

    model = ResNet50(weights='imagenet', include_top=True, input_tensor=x)
    return model


def load_symmetric():

    real_paths = glob(real_images+'/*.JPEG')
    real_paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_adv = glob(adv_dir+'/*.png')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f)))

    x_real = np.empty((len(real_paths), 224, 224, 3))
    reals = load_data.load_dir(real_paths, arr=x_real, start=0,
                               end=len(real_paths))

    x_adv = np.empty((len(paths_adv), 224, 224, 3))
    x_adv = load_data.load_dir(paths_adv, arr=x_adv, start=0,
                               end=len(paths_adv))

    reals = preprocess_input(reals)
    adv = preprocess_input(x_adv)

    return reals, adv


def load_fcn():
    pass


def test_pool(model, preds, data, topk, pool='avg'):

    preds_avg = predictions(model, data, topk=topk)
    recovered = 0.
    for i, (p_real, p_rec) in enumerate(zip(preds, preds)):
        if topk > 1:
            if set(p_real) <= set(p_rec):
                recovered += 1
        else:
            if p_real == p_rec:
                recovered += 1
            else:
                print p_real, p_rec

    print "recovered top{} labeled examples: {}".format(topk, recovered)
    print "top{} recovery rate: {}".format(topk, recovered/len(preds))

    """
    for x in p_real:
        if x not in p_rec:
            print x
            plt.imshow(real_good[i])
            plt.show()
            plt.imshow(adv_good[i])
            plt.show()
            plt.imshow(adv_bilateral[i])
            plt.show()
            break
    # print p_rec, p_real
    """


def test_bilateral(model, preds, data, f, s, topk):

    adv_bilateral = bilateral(data, f, s, s)
    preds_bilateral = predictions(model, adv_bilateral, topk=topk)

    recovered = 0.
    for i, (p_real, p_rec) in enumerate(zip(preds, preds_bilateral)):
        if topk > 1:
            if set(p_real) <= set(p_rec):
                recovered += 1
        else:
            if p_real == p_rec:
                recovered += 1

    print "recovered top{} labels : {}".format(topk, recovered)
    print "top{} recovery rate: {}".format(topk, recovered/len(preds))


if __name__ == '__main__':

    real, adv = load_symmetric()
    model = load_vanilla_model()
    """
    # test
    print np.argmax(model.predict(np.expand_dims(real[0], 0)))
    print np.argmax(model.predict(np.expand_dims(adv[0], 0)))
    sys.exit(0)
    """
    preds = predictions(model, real)
    preds2 = predictions(model, adv)
    diff = 0
    idxs = []
    # vanilla differences
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            idxs.append(i)

    assert (len(idxs) + diff) == len(preds)

    print "preliminary model difference on set: {} adversarials".format(diff)

    real_good = np.array([real[i] for i in range(len(real)) if i not in idxs])
    adv_good = np.array([adv[i] for i in range(len(adv)) if i not in idxs])

    assert (len(real_good) == diff)

    preds = predictions(model, real_good)
    preds_adv = predictions(model, adv_good)
    preds5 = predictions(model, real_good, proba=False, topk=5)
    preds10 = predictions(model, real_good, proba=False, topk=10)

    diff = 0
    bad_idx = []
    for i, (p_r, p_adv) in enumerate(zip(preds, preds_adv)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)
    assert len(bad_idx) == 0

    # for x in real_good:
    #     print np.max(model.predict(np.expand_dims(x, 0)))

    # model_avg = load_recovery_model(bilateral=False)
    # The predictions passed in MUST correspond to the topk given
    # test_pool(model_avg, preds, adv_good, topk=1)

    filters = [2, 3, 4, 5, 7]
    sigma_size = [25, 35, 50]
    for f in filters:
        for s in sigma_size:
            print "f: {}, sc: {}, ss: {}".format(f, s, s)
            model_avg, model = load_recovery_model(bilateral=True)
            test_bilateral(model, preds, adv_good, f, s, topk=1)
