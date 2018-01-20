import os
import gc
import sys
import cv2
import load_data
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from pprint import pprint
from imageio import imwrite
from collections import defaultdict
from scipy.misc import imresize, imshow

# local imports
import tf_models
from srgan import superres
from cw_inception_wrapper import InceptionModel

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

adv_root = '../../../images/adversarials'
model_loaded = False
BATCH_SIZE = 32

def load_args():

    parser = argparse.ArgumentParser(description='tools for recovering the image beneath an adversarial example')
    parser.add_argument('-m', '--model', default='inception_v4', type=str)
    parser.add_argument('-a', '--attack', default='fgsm', type=str, help='directory of adversarial images')
    parser.add_argument('-d', '--dataset', default='imagenet', type=str, help='dataset to recover from')
    parser.add_argument('-s', '--save_dir', type=str)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    args = parser.parse_args()
    return args


def predictions_tf(graph, sess, model, samples, topk=0):
    if BATCH_SIZE > 1:
        return predictions_tf_batch(graph, sess, model, samples, topk)
    else:
        return predictions_tf_single(graph, sess, model, samples, topk)


def predictions_tf_single(graph, sess, model, samples, topk=0):
    with graph.as_default():
        samples = samples.astype(np.float32)
        iset = np.zeros((len(samples), topk))
        for sample in samples:
            x = np.expand_dims(sample, 0)
            probabilities = model.predict(x, sess)
            if topk > 0:
                preds = np.argsort(probabilities)[::-1]
                iset[j] = preds[:topk]

        predictions = iset
    return predictions


def predictions_tf_batch(graph, sess, model, samples, topk=0):
    with graph.as_default():
        batch_size = BATCH_SIZE
        samples = samples.astype(np.float32)
        iset = np.zeros((len(samples), topk))
        x = tf.placeholder(tf.float32, shape=(batch_size, 299, 299, 3))
        last_batch_size = len(samples) % batch_size
        if last_batch_size != 0:
            num_batches = int(len(samples) / batch_size) + 1
        else:
            num_batches = len(samples) / batch_size

        for i in range(num_batches):
        #for sample in samples:
            n = batch_size * i
            m = batch_size * (i + 1)
            if i == num_batches - 1:
                small_batch = samples[n:]
            else:
                small_batch = samples[n:m]
            probabilities = model.predict(small_batch, sess)
            if topk > 0:
                for j, pred in enumerate(probabilities):
                    preds = np.argsort(pred)[::-1]
                    iset[j+n] = preds[:topk]
        
        predictions = iset
    return predictions


def load_model(graph, sess, m_name, n_classes, cw=False):

    with graph.as_default():
        if cw is True:
            model = InceptionModel(sess)   
        else:
            if m_name == 'inception_v3':
                model = tf_models.InceptionV3Model(sess) 
                model._build()
            elif m_name == 'inception_v4':
                model = tf_models.InceptionV4Model(sess)
                model._build()
            elif m_name == 'inception_resnet_v2':
                model = tf_models.InceptionResNetModel(sess)
                model._build()
            elif m_name == 'resnet_v2_101':
                model = tf_models.ResNetV2Model(sess)
                model._build()
            else:
                raise ValueError('Model unsupported: {}'
                                 'choose "inception_v3", "inception_v4", "resnet_v2",'
                                 'inception_resnet_v2"'.format(m_name))
    return model


def load_path_npy(paths, arr, start=0, end=0):

    assert arr.ndim == 4
    imshape = (arr.shape[1], arr.shape[2], arr.shape[3])
    for idx, i in enumerate(range(start, end)):
        image = np.load(paths[idx])
        arr[i] = image
    print "Loaded {} images".format(len(paths))

    return arr


def load_npy(real, adv, n, shape=299):
    if n is None:
        n = len(real) - 1
    paths_real = glob(real+'_npy/*.npy')
    print paths_real[0]
    print paths_real[1][73:76]
    # paths_real.sort(key=lambda f: int(filter(str.isdigit, f[73:76])))
    paths_real.sort(key=lambda f: int(filter(str.isdigit, f[78:88])))
    paths_adv = glob(adv+'_npy/*.npy')
    paths_adv.sort(key=lambda f: int(filter(str.isdigit, f[77:87])))
    # paths_adv.sort(key=lambda f: int(filter(str.isdigit, f[72:85])))
    paths_real = paths_real[:n]
    paths_adv = paths_adv[:n]
    x_real = np.empty((len(paths_real), shape, shape, 3))
    real = load_path_npy(paths_real, arr=x_real, start=0, end=len(paths_real))
    x_adv = np.empty((len(paths_adv), shape, shape, 3))
    adv = load_path_npy(paths_adv, arr=x_adv, start=0, end=len(paths_adv))
    return real, adv


def load_images(attack_str, data_str, net_str, n=10000):
    
    if data_str == 'imagenet':
        dataset = '/imagenet/symmetric'
    elif data_str == 'cifar':
        dataset = '/cifar/symmetric'
    elif data_str == 'mnist':
        dataset = '/mnist/symmetric'
    else: 
        raise ValueError("Only Imagenet, Cifar, and MNIST supported. Got {}"
                         .format(data_str))

    base = adv_root+'/'+attack_str+dataset+'/'+net_str+'/'
    if attack_str == 'fgsm':
        real_dir = base+'real'
        adv_dir = base+'adv'
    
    elif attack_str == 'lbfgs':
        real_dir = base+'real'
        adv_dir = base+'adv'

    elif attack_str == 'deepfool':
        real_dir = base+'real'
        adv_dir = base+'adv'

    elif attack_str == 'mim':
        real_dir = base+'real'
        adv_dir = base+'adv'

    elif attack_str == 'cw':
        real_dir = base+'real'
        adv_dir = base+'adv'

    real, adv = load_npy(real_dir, adv_dir, n, 299)
    return (real, adv)


def bilateral(arr, f, sc, ss):

    assert len(arr) > 0 and arr.ndim == 4
    arr = arr.astype(np.float32)
    filtered = np.empty(arr.shape)
    for i, image in enumerate(arr):
        filtered[i] = cv2.bilateralFilter(image, f, sc, ss)
    return filtered


def pyramidal_bf(arr):

    assert len(arr) > 0 and arr.ndim == 4
    arr = arr.astype(np.float32)
    filtered = np.empty(arr.shape)
    for i, image in enumerate(arr):
        z = cv2.bilateralFilter(image, 3, 25, 25)
        imshow(z)
        imshow(image - z)
        #zx = cv2.bilateralFilter(z, 3, 10, 10)
        # super_zx = superres(image)[0]
        # imshow(super_zx)
        #szx = imresize(super_zx, (299, 299, 3))
        #szx = szx / 255. - 0.5
        filtered[i] = z
        

    return filtered

""" 
should return percentage of images that can be recovered 
with a bilateral filter, using a grid search across:
kernel, color range, space range
"""
def search_params(x, y, model, max_depth, level=1):

    kernel_range = [1, 2, 3, 5, 7, 9, 15, 20, 25, 30, 40, 50, 60]
    sigma_color_range = [1, 3, 5, 10, 15, 25, 40, 50, 75, 100, 150]
    sigma_space_range = [1, 3, 5, 10, 15, 25, 40, 50, 75, 100, 150]
    
    def search(params, level):
        if level > max_depth:
            return params
        for k_w in kernel_range:
            for s_c in sigma_color_range:
                for s_s in sigma_space_range:
                    z = cv2.bilateralFilter(x, k_w, s_c, s_s)
                    pred = np.argmax(model.predict(np.expand_dims(z, 0)))
                    predy = np.argmax(model.predict(np.expand_dims(y, 0)))
                    if (pred >= predy-1) and (pred <= predy + 1):
                        zy = cv2.bilateralFilter(y, k_w, s_c, s_s)
                        norm = np.linalg.norm(z - zy)
                        return (k_w, s_c, s_s, norm)
                    else:
                        res = search((None, None, None, None), level+1)

    params = search((None, None, None, None), level)
    if params is None:
        params = (None, None, None, None)
    return params

def test_bf_params(real, adv, model, sess):
    hits = 0.
    start = 0
    real = real[start:]
    adv = adv[start:]
    param_count = defaultdict(int)
    params = np.zeros((len(adv), 3))
    for i, (r, a) in enumerate(zip(real, adv)):
        i = i + start
        print "Image {}".format(i)
        r = r.astype(np.float32)
        a = a.astype(np.float32)
        """return smallest params because we perfer to retain information """
        res = search_params(a, r, model, max_depth=1)
        if res[0] is None:
            res = np.array([0, 0, 0])
            print "No workable params within search space" 
        else:
            hits += 1
            k, sc, ss, norm = res
            res = np.array(res[:3])
            param_str = "{}-{}-{}".format(str(k), str(sc), str(ss))
            param_count[param_str] += 1
            print "Found params for image {}: ({}, {}, {}), norm: {}".format(i, k, sc, ss, norm)

        params[i] = res
        print "saved ", params[i]
        print "---------------------------------------------"

    print "Found params for: {}% of images".format(hits/(len(adv)-start))
    print "Full paramater count: "
    pprint(dict(param_count))
    np.save(model._name+'_fgsm_500_params', params)
    sys.exit(0)
 

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


def show(real, adv, model=None, sess=None, i=None):
    import matplotlib.pyplot as plt
    if model is None:
        import scipy.misc
        scipy.misc.imshow(real)
        scipy.misc.imshow(adv)
        return
    pred_real = np.argmax(model.predict(np.expand_dims(real, 0), sess))
    pred_adv = np.argmax(model.predict(np.expand_dims(adv, 0), sess))
    norm = np.linalg.norm(real - adv)
    real = (((real + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
    adv = (((adv + 1.0) * 0.5) * 255.0)#.astype(np.uint8)
    plt.figure()
    if i is not None:
        plt.suptitle("Example {}".format(str(i)))
    ax = plt.subplot(1, 3, 1)
    ax.set_title("real: "+str(pred_real))
    plt.imshow(real)
    ax = plt.subplot(1, 3, 2)
    ax.set_title("adv: "+str(pred_adv))
    plt.imshow(adv)
    ax = plt.subplot(1, 3, 3)
    ax.set_title("diff: "+str(norm))
    plt.imshow(real - adv)
    plt.show()


""" Measures top1 and top5 recovery rates """
def rate(preds_1, preds_5):
    
    preds_1adv, preds_1bf = preds_1
    preds_5adv, preds_5bf = preds_5

    misses = []
    """ top1 """
    recovered_examples_1 = 0.
    for i, (p_real, p_rec) in enumerate(zip(preds_1adv, preds_1bf)):
        if p_real == p_rec:
            recovered_examples_1 += 1
        else:
            misses.append(i)
        
    rate_1 = recovered_examples_1 / len(preds)

    """ top5 """
    recovered_examples_5 = 0.
    for i, (p_real, p_rec) in enumerate(zip(preds_1adv, preds_5bf)):
        if set(p_rec) <= set(p_real):
            recovered_examples_5 += 1
        else:
            pass
    rate_5 = recovered_examples_5 / len(preds)
    return (rate_1, rate_5), misses


if __name__ == '__main__':

    args = load_args()
    real, adv = load_images(args.attack, args.dataset, args.model, n=500)
    if args.dataset == 'imagenet': n_classes = 1001
    if args.dataset == 'cifar': n_classes = 10
    if args.dataset == 'mnist': n_classes = 10
    if args.attack == 'cw': cw = True
    else: cw = False
    tf.set_random_seed(1234)
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    model = load_model(graph, sess, args.model, n_classes)

    test_bf_params(real, adv, model, sess)
    # vanilla differences
    # weed out non adversarials
    diff = 0
   
    preds = predictions_tf(graph, sess, model, real, 1)
    preds2 = predictions_tf(graph, sess, model, adv, 1)

    bad_idx = []
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)
    print "preliminary model difference on set: {} true adversarials".format(diff)

    real_good = np.array([real[i] for i in range(len(real)) if i not in bad_idx])
    adv_good = np.array([adv[i] for i in range(len(adv)) if i not in bad_idx])
    preds = predictions_tf(graph, sess, model, real_good, 1)
    preds2 = predictions_tf(graph, sess, model, adv_good, 1)
    preds5 = predictions_tf(graph, sess, model, real_good, 3)
    preds10 = predictions_tf(graph, sess, model, real_good, 5)

    diff = 0
    bad_idx = []
    for i, (p_r, p_adv) in enumerate(zip(preds, preds2)):
        if p_r != p_adv:
            diff += 1
        else:
            bad_idx.append(i)
    assert len(bad_idx) == 0

    filters = [3, 5, 9, 11, 13, 15]
    sigma_s = [25, 30, 35, 45, 55, 65]
    sigma_c = [25, 30, 35, 45, 55, 65]
    max_kernel = 3
    max_sigma = 25
    best_rate = 0

    adv_bilateral = pyramidal_bf(adv_good)
    preds_rec = predictions_tf(graph, sess, model, adv_bilateral, 1)
    preds_rec_5 = predictions_tf(graph, sess, model, adv_bilateral, 4)

    (rate_1, rate_5), misses = rate((preds, preds_rec), (preds5, preds_rec_5))
    
    for miss in misses:
        show(real_good[miss], adv_bilateral[miss], model, sess, miss)

    n_recovered_1 = rate_1 * len(preds)
    n_recovered_5 = rate_5 * len(preds)
    
    print "recovered top1 labeled examples: {}".format(n_recovered_1)
    print "top1 recovery rate: {}".format(rate_1)
    print "recovered top5 within top10 labeled examples: {}".format(n_recovered_5)
    print "top5 recovery rate: {}".format(rate_5)
    print "-----------------------------------------------------------"
    sys.exit(0)
    


    for f in filters:
        for s in sigma_s:
            c = s
            adv_bilateral = 0.
            adv_bilateral = bilateral(adv_good, f, c, s)

            preds_rec = predictions_tf(graph, sess, model, adv_bilateral, 1)
            preds_rec_5 = predictions_tf(graph, sess, model, adv_bilateral, 4)
            
            # adv_avg = predictions(model_avg, adv_good, proba=True)[:10]
            # preds_rec = predictions(model, adv_avg, prep=True)
            # preds_rec_10 = predictions(model, adv_avg, topk=10, prep=True)

            rate_1, rate_5 = rate((preds, preds_rec), (preds5, preds_rec_5))
            n_recovered_1 = rate_1 * len(preds)
            n_recovered_5 = rate_5 * len(preds)
            print "kernel width: {}, color range: {}, spatial range: {}".format(f, c, s)
            print "recovered top1 labeled examples: {}".format(n_recovered_1)
            print "top1 recovery rate: {}".format(rate_1)
            print "recovered top5 within top10 labeled examples: {}".format(n_recovered_5)
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
        bilateral_save(model, adv_good, real_good, max_kernel, max_sigma)

