import sys
import cv2
import gcnn
import load_data
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from glob import glob
from imageio import imwrite
from ResNet50 import ResNet50
from keras.layers import Input
from keras.models import Sequential
# from fcn_models import DenseNet_FCN
# from fcn_models import FCN_ResNet50
from keras.layers.pooling import AveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions

adv_weights = '../models/weights/detectors/lbfgs/imagenet/iter_0.h5'
# adv_dir = '../../images/adversarials/lbfgs/imagenet/symmetric/test/adv'
# adv_dir = '/home/neale/tmp/bilateral_lbfgs'
adv_dir = '/home/neale/tmp/bilateral_fgsm'
# real_images = '../../images/adversarials/lbfgs/imagenet/symmetric/test/real'
# real_images = '/home/neale/tmp/bilateral_real'
real_images = '/home/neale/tmp/bilateral_fgsm_real'
# real_images = '../../images/imagenet12/recovery_set_all'


def predictions(model, x, proba=False, topk=0, prep=False):

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
        iset = []
        for im in x:
            im = np.expand_dims(im, 0)
            if topk > 0:
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
    x = AveragePooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    if bilateral is True:
        avg_model = Model(inputs=inputs, outputs=x)
        model = ResNet50(weights='imagenet', include_top=True)
        return (avg_model, model)

    model = ResNet50(weights='imagenet', include_top=True, input_tensor=x)
    return model


def load_symmetric():
    real_paths = glob(real_images+'/*.png')
    real_paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths_adv = glob(adv_dir+'/*.png')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f)))

    x_real = np.empty((len(real_paths), 224, 224, 3))
    reals = load_data.load_dir(real_paths, arr=x_real, start=0, end=len(real_paths))

    x_adv = np.empty((len(paths_adv), 224, 224, 3))
    x_adv = load_data.load_dir(paths_adv, arr=x_adv, start=0, end=len(paths_adv))

    """
    reals = preprocess_input(reals)
    adv = preprocess_input(x_adv)

    return reals, adv
    """
    return reals, x_adv


def load_fcn():
    pass


def bilateral_all_the_things(model, adv, real):

    kernel = 3
    sigma = 35
    adv_bilateral = bilateral(adv_good, kernel, sigma, sigma)
    for i in range(len(adv_bilateral)):
        imwrite('/home/neale/tmp/bilateral_fgsm/im_{}.png'.format(i), adv_bilateral[i])
        imwrite('/home/neale/tmp/bilateral_fgsm_real/im_{}.png'.format(i), real[i])

    sys.exit(0)


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

    real, adv = load_symmetric()
    model = load_vanilla_model()
    build_resistance_dict(model, real, adv)

    """
    # test
    print np.argmax(model.predict(np.expand_dims(real[0], 0)))
    print np.argmax(model.predict(np.expand_dims(adv[0], 0)))
    sys.exit(0)
    """
    from scipy.misc import imshow

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
    top10 = predictions(model, real_good, proba=False, topk=10, prep=True)
    top5 = predictions(model, real_good, proba=False, topk=5, prep=True)

    diff = 0
    bad_idx = []
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)
    assert len(bad_idx) == 0

    bilateral_all_the_things(model, adv_good, real_good)
    sys.exit(0)
    # for x in real_good:
    #     print np.max(model.predict(np.expand_dims(x, 0)))

    filters = [3, 4, 5, 7]
    sigma_size = [25, 30, 35, 40]
    for f in filters:
        for s in sigma_size:
            print "f: {}, sc: {}, ss: {}".format(f, s, s)
            model_avg, model = load_recovery_model(bilateral=True)
            adv_good = adv_good.astype(np.uint8)
            # adv_avg = predictions(model_avg, adv_good, proba=True)[:10]
            adv_bilateral = bilateral(adv_good, f, s, s)
            """
            for i in range(5):
                # plt.imshow(adv_good[i].astype(np.uint8))
                # plt.show()
                plt.imshow(adv_avg[i].astype(np.uint8))
                plt.show()
                imwrite('/home/neale/tmp/bilateral_lbfgs/avg_recovered_lbfgs_{}_{}.png'.format(i, f), adv_avg[i].astype(np.uint8))
                imwrite('/home/neale/tmp/save/lbfgs_{}_{}.png'.format(i, f), adv_good[i])
            """
            # preds_rec = predictions(model, adv_avg, prep=True)
            preds_rec = predictions(model, adv_bilateral, prep=True)
            # preds_rec_10 = predictions(model, adv_avg, topk=10, prep=True)
            preds_rec_10 = predictions(model, adv_bilateral, topk=10, prep=True)

            # for x in adv_bilateral:
            #     print np.max(model.predict(np.expand_dims(x, 0)))

            recovered_examples = 0
            for i, (p_real, p_rec) in enumerate(zip(preds, preds_rec)):
                if p_real == p_rec:
                    recovered_examples += 1

            print "recovered top1 labeled examples: {}".format(recovered_examples)
            print "top 1recovery rate: {}".format(float(recovered_examples)/len(preds))

            recovered_examples = 0
            for i, (p_real, p_rec) in enumerate(zip(top5, preds_rec_10)):
                if set(p_real) <= set(p_rec):
                    recovered_examples += 1
                else:
                    pass
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
            print "recovered top5 within top10 labeled examples: {}".format(recovered_examples)
            print "top5 recovery rate: {}".format(float(recovered_examples)/len(preds))
