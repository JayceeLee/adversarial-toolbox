import os
import gc
import sys
import cv2
import load_data
import argparse
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from glob import glob
from imageio import imwrite
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from scipy.misc import imresize
#from keras.applications.resnet50 import preprocess_input
#from keras.applications.resnet50 import decode_predictions

adv_weights = '../models/weights/detectors/lbfgs/imagenet/iter_0.h5'
adv_dir = '../../../images/adversarials/cw/imagenet/symmetric/l2/inception_v3/adv'
real_dir = '../../../images/adversarials/cw/imagenet/symmetric/l2/inception_v3/real'

save_adv = '../../../images/adversarials/fgsm/imagenet/symmetric/bilateral_adv'
save_real = '../../../images/adversarials/fgsm/imagenet/symmetric/bilateral_real'

def load_args():

    parser = argparse.ArgumentParser(description='tools for recovering the image beneath an adversarial example')
    parser.add_argument('-r', '--real_dir', default=real_dir, type=str, help='directory where PNG images are stored')
    parser.add_argument('-a', '--adv_dir', default=adv_dir, type=str, help='directory of adversarial images')
    args = parser.parse_args()
    return args


def predictions(model, x, proba=False, topk=1, prep=False):

    assert len(x) > 0

    if prep:
        x = preprocess_input(x.astype(np.float32))
    if proba:
        iset = np.empty((len(x), 224, 224, 3))
        for i, im in enumerate(x):
            im = np.expand_dims(im, 0)
            proba = model.predict(im)
            iset[i] = proba
    else:
        iset = np.empty((len(x), topk))
        for i, im in enumerate(x):
            im = np.expand_dims(im, 0)
            if topk > 0:
                preds = model.predict(im)
                labels = np.argsort(-preds)[0][:topk]
                iset[i] = labels
            else:
                label = np.argmax(model.predict(im))
                iset[i] = label

    return iset


def load_vanilla_model():

    # return ResNet50(weights='imagenet', include_top=True)
    return InceptionV3(weights='imagenet', include_top=True)

"""
def load_adv_gcnn(weights):

    base = gcnn.load_base_gcnn()
    model = ResNet50(weights=None, include_top=False, input_tensor=base)
    model = gcnn.load_clf(model)
    model.load_weights(weights)

    return model
"""

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
    x = AveragePooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    if bilateral is True:
        avg_model = Model(inputs=inputs, outputs=x)
        model = InceptionV3(weights='imagenet', include_top=True)
        return (avg_model, model)

    model = InceptionV3(weights='imagenet', include_top=True, input_tensor=x)
    return model


def load_path_npy(paths, arr, start=0, end=0):

    assert arr.ndim == 4
    imshape = (arr.shape[1], arr.shape[2], arr.shape[3])
    for idx, i in enumerate(range(start, end)):
        image = np.load(paths[idx])[0]
        image = imresize(image, imshape)
        arr[i] = image
    print "Loaded {} images".format(len(paths))

    return arr


def load_npy(real, adv, n, shape=299):
    if n is None:
        n = len(real) - 1
    print real
    paths_real = glob(real+'/*.npy')
    print paths_real
    paths_real.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_adv = glob(adv+'/*.npy')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_real = paths_real[-n:]
    paths_adv = paths_adv[-n:]
    print paths_real
    x_real = np.empty((len(paths_real), shape, shape, 3))
    real = load_path_npy(paths_real, arr=x_real, start=0, end=len(paths_real))
    x_adv = np.empty((len(paths_adv), shape, shape, 3))
    adv = load_path_npy(paths_adv, arr=x_adv, start=0, end=len(paths_adv))
    return real, adv


def load_symmetric(real, adv, n_images=None):
    if n_images is None:
        n_images = len(real) - 1
    paths_real = glob(real+'/*.png')
    paths_real.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_adv = glob(adv+'/*.png')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f)))

    paths_real = paths_real[-n_images:]
    paths_adv = paths_adv[-n_images:]
    print paths_real[0]
    print paths_adv[0]

    x_real = np.empty((len(paths_real), 224, 224, 3))
    real = load_data.load_dir(paths_real, arr=x_real, start=0, end=len(paths_real))

    x_adv = np.empty((len(paths_adv), 224, 224, 3))
    adv = load_data.load_dir(paths_adv, arr=x_adv, start=0, end=len(paths_adv))

    return real, adv


def bilateral_all_the_things(model, adv, real, k, s):

    adv_bilateral = bilateral(adv_good, k, s, s)
    if not os.path.exists(save_adv):
            os.makedirs(save_adv)
    if not os.path.exists(save_real):
            os.makedirs(save_real)
    for i in range(len(adv_bilateral)):
        imwrite(save_adv+'/im_{}.png'.format(i), adv_bilateral[i])
        imwrite(save_real+'/im_{}.png'.format(i), real[i])


def get_norm(real, adv):

    if real.shape != adv.shape:
        raise ValueError("Both data sets must have the same shape, got {} and {}"
                         .format(real.shape, adv.shape))
    diff = real - adv
    if diff.ndim == 3:
        l = np.linalg.norm(diff)
    if diff.ndim == 4:
        l = 0.
        for x in diff:
            l += np.linalg.norm(x)
        l /= len(diff)
    return l


def build_resistance_dict(model, real, adv):

    """
    collisions dictionary tracks a labels resistance to BF
    real label: [name, #attempts, #successful BF, [new conf]]
    """
    success = 0.
    collisions = {}
    # real = real[:100]
    # adv = adv[:100]
    real = preprocess_input(real.astype(np.float32))
    adv = preprocess_input(adv.astype(np.float32))
    preds_real = model.predict(real)
    preds_adv = model.predict(adv)
    name_real = decode_predictions(preds_real, top=1)
    name_adv = decode_predictions(preds_adv, top=1)
    for i, (pr, pa) in enumerate(zip(preds_real, preds_adv)):
        label_real = np.argmax(pr)
        label_adv = np.argmax(pa)
        stats_real = name_real[i][0]
        stats_adv = name_adv[i][0]
        if label_real != label_adv:  # resisted
            if str(stats_real[1]) not in collisions:
                collisions[str(stats_real[1])] = [label_real, 1, 0, [stats_adv[-1]], [stats_real[-1]]]
            else:
                collisions[str(stats_real[1])][1] += 1
                collisions[str(stats_real[1])][-2].append(stats_adv[-1])
                collisions[str(stats_real[1])][-1].append(stats_real[-1])
        else:
            success += 1
            if str(stats_real[1]) not in collisions:
                collisions[str(stats_real[1])] = [label_real, 1, 1, [stats_adv[-1]], [stats_real[-1]]]
            else:
                collisions[str(stats_real[1])][1] += 1
                collisions[str(stats_real[1])][2] += 1
                collisions[str(stats_real[1])][-2].append(stats_adv[-1])
                collisions[str(stats_real[1])][-1].append(stats_real[-1])

    print success / len(real)
    with open('/home/neale/BF.txt', 'wb') as f:
        for (k, v) in collisions.items():
            f.write(str(k) + ' : ' + str(v) + '\n')

    o = OrderedDict(sorted(collisions.items(), key=lambda x: x[1][1] - x[1][2]))
    count = 0
    same = 0.
    total = 0.
    for (k, v) in o.items():
        if v[1] == v[2]:
            same += 1
        total += v[2]
        count += 1
        conf_diff = np.mean(v[-1]) - np.mean(v[-2])
        print k, " : ", v, conf_diff
    print same/count
    print "recovered: {} / {} ".format(total, len(real))
    sys.exit(0)


if __name__ == '__main__':

    args = load_args()
    # real, adv = load_symmetric(args.real_dir, args.adv_dir, 100)
    real, adv = load_npy(args.real_dir, args.adv_dir, 100)
    model = load_vanilla_model()

    preds = predictions(model, real, prep=True)
    preds2 = predictions(model, adv, prep=True)
    diff = 0
    bad_idx = []
    # vanilla differences
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)

    assert (len(bad_idx) + diff) == len(preds)

    print "preliminary model difference on set: {} true adversarials".format(diff)

    real_good = np.array([real[i] for i in range(len(real)) if i not in bad_idx])
    adv_good = np.array([adv[i] for i in range(len(adv)) if i not in bad_idx])

    assert (len(real_good) == diff)

    preds = predictions(model, real_good, prep=True)
    preds2 = predictions(model, adv_good, prep=True)

    top10 = predictions(model, real_good, proba=False, topk=5, prep=True)
    top5 = predictions(model, real_good, proba=False, topk=2, prep=True)
    print top10.shape
    print top5.shape
    diff = 0
    bad_idx = []
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)
    assert len(bad_idx) == 0
    gc.collect()

    # print get_norm(real_good, adv_good)

    filters = [2, 3, 5, 9, 11]
    sigma_s = [15, 25, 30, 35, 45]
    sigma_c = [15, 25, 30, 35, 45]
    max_kernel = 3
    max_sigma = 25
    best_rate = 0
    model_avg, model = load_recovery_model(bilateral=True)
    for f in filters:
        for s in sigma_s:
            for c in sigma_c:
                adv_bilateral = 0.
                adv_bilateral = bilateral(adv_good, f, s, s)

                preds_rec = predictions(model, adv_bilateral, prep=True)
                preds_rec_10 = predictions(model, adv_bilateral, topk=10, prep=True)

                # adv_avg = predictions(model_avg, adv_good, proba=True)[:10]
                # preds_rec = predictions(model, adv_avg, prep=True)
                # preds_rec_10 = predictions(model, adv_avg, topk=10, prep=True)

                recovered_examples_1 = 0.
                for i, (p_real, p_rec) in enumerate(zip(preds, preds_rec)):
                    if p_real == p_rec:
                        recovered_examples_1 += 1
                rate_1 = recovered_examples_1 / len(preds)

                recovered_examples_5 = 0.
                for i, (p_real, p_rec) in enumerate(zip(top5, preds_rec_10)):
                    if set(p_real) <= set(p_rec):
                        recovered_examples_5 += 1
                    else:
                        pass
                        """
                        a = preprocess_input(np.expand_dims(adv_bilateral[i], 0).astype(np.float32))
                        r = preprocess_input(np.expand_dims(real_good[i], 0).astype(np.float32))
                        plt.figure()
                        plt.subplot(1, 2, 1)
                        plt.imshow(r[0])
                        plt.subplot(1, 2, 2)
                        plt.imshow(a[0])
                        plt.suptitle("Could not recover")
                        plt.show()
                        """
                rate_5 = recovered_examples_5 / len(preds)

                print "kernel width: {}, color range: {}, spatial range: {}".format(f, c, s)
                print "recovered top1 labeled examples: {}".format(recovered_examples_1)
                print "top1 recovery rate: {}".format(rate_1)
                print "recovered top5 within top10 labeled examples: {}".format(recovered_examples_5)
                print "top5 recovery rate: {}".format(rate_5)
                print "-----------------------------------------------------------"
                if rate_5 > best_rate:
                    best_rate = rate_5
                    max_sigma = s
                    max_kernel = f

    print "* max recovery: {}".format(best_rate)
    print "* max kernel width: {}".format(max_kernel)
    print "* max signal range: {}".format(max_sigma)

    if best_rate >= 0.90:
        bilateral_all_the_things(model, adv_good, real_good, max_kernel, max_sigma)

