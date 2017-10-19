import os
import sys
import keras
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import train_detector as td
import gsvm
import gcnn
import gsvm as gs
from attacks import lbfgs
from scipy.misc import imsave, imread
from glob import glob
from models import cifar_model
import load_data
from sklearn.externals import joblib
from sklearn.utils import shuffle
import cifar_base
from keras.models import Model
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

home = '/home/neale/repos/adversarial-toolbox'
save_dir_base = home+'/images/adversarials/lbfgs/imagenet/test1/resnet_test'
images_dir = home+'/images/imagenet12/train/'
wpath_base = home+'/toolbox/models/weights/detectors/lbfgs/imagenet/iter_'
wpath_init = 'iter0'
start_iter = 0
min_nonzero = 1000
log_file = './cnn_scores.txt'
LOG = True
n_images = 1000


def logger(data):

    with open(log_file, 'a+b') as f:
        f.write(data)


def validate_label(model, x):
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = preprocess_input(x)
    preds = model.predict(x)
    pred = decode_predictions(preds, top=1)
    return pred


def generate_adv(wpath, it):

    num_gen = 0
    start = 0
    if it == 0:
        targets = 1000
        top = 'vanilla'
    else:
        targets = 2
        top = 'detector'

    train = load_data.load_real_resnet(n_images)
    print "loading from ", wpath
    # kmodel = gcnn.load_resnet(top=top, weight_path=wpath, gcnn=False)
    kmodel = ResNet50(weights='imagenet', include_top=True)
    model = lbfgs.load_model(model=kmodel)
    kmodel.summary()
    val_model = ResNet50(weights='imagenet')

    for i, img in enumerate(train[start:n_images]):
        #lbfgs.display_im(img)
        x = img+0.
        init_pred = validate_label(val_model, x)
        print init_pred
        adv_obj = lbfgs.generate(model, img[:, :, ::-1], targets=targets)
        if adv_obj is None:
            print "Failed to apply attack"
            continue
        adv_img = adv_obj.image
        if type(adv_img) is None:
            print "FAILED, unsuccessful attack"
            continue
        if min_nonzero and (np.count_nonzero(adv_img) < min_nonzero):
            print "FAILED, too many zero pixels"
            continue
        res = lbfgs.print_stats(adv_img, adv_obj.original_image, model)
        if res == 1:
            print "FAILED with img {}".format(i)
            print "------------------------"
            continue

        num_gen += 1
        res_dir = save_dir_base+'{}/'.format(str(it))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        save_str = res_dir+'im_{}.png'.format(num_gen)

        # k validation
        x = adv_obj.image[:, :, ::-1]
        y = np.expand_dims(x, axis=0)
        y = preprocess_input(y)
        preds = val_model.predict(y)
        kpred = np.argmax(preds)
        print "keras pred: ", kpred
	imsave(save_str, x)
        print "SUCCESS, images saved {}".format(num_gen)
        print "Images attempted for this run: {}".format(i+1)
        print "------------------------"
    if LOG:
        logger('generated {} images with L-BFGS\n'.format(num_gen))

    return num_gen


def test_model(model):

    x = load_data.load_real_resnet(5000)
    preds = model.predict(x, batch_size=32)
    print preds


def ft_detector(it, train, test, wpath, old_wpath):

    if it > 0:
        top = 'detector'
    else:
        top = 'vanilla'

    # model = gcnn.load_test()
    # gcnn.test_gcnn(train, test, model)

    if wpath[-3:] != '.h5':
        wpath += '.h5'
    model = gcnn.load_resnet(top, wpath, gcnn=True)
    # model = ResNet50(weights='imagenet', include_top=True)
    model, hist = gcnn.train_gcnn(train, test, model)

    acc = hist[-1]
    path = wpath + 'grad.h5'  # + str(acc)[:4] + '.h5'
    model.save_weights(path)
    print "saving weights to {}".format(path)
    if LOG:
        logger('fine tuned CNN to {}% accuracy\n'.format(acc*100))
    # test_model(model)
    return acc, model


def train_base():

    model = cifar_base.train()
    model = ResNet50(weights='imagenet', include_top=True)
    #model, hist = gcnn.train_gcnn(train, test, model)
    model.save_weights(wpath_base+wpath_init)


def cascade_classifier(train, test):

    names = []
    model = ResNet50(weights='imagenet', include_top=True)
    for layer in model.layers:
        if (('res' in layer.name or 'conv' in layer.name) and
           ('bn' not in layer.name) and
           (layer.name[-2:] == '2b')):

            if hasattr(model.get_layer(layer.name), 'output'):
                names.append(layer.name)

    # feed the image to get features at the specified layer
    x_t, y_t = train
    x_v, y_v = test
    x_t = x_t[:1000]
    x_v = x_v[:500]
    y_t = y_t[:1000]
    y_v = y_v[:500]

    accuracy = 0.
    x_v_real = np.array([])
    y_v_real = np.array([])
    n_examples = len(x_t)

    labels = np.argmax(y_v, axis=1)
    adv_idxs = np.where(labels == 0)[0]
    total_adv = float(len(adv_idxs))
    for i, name in enumerate(names):

        x_ts = np.zeros((n_examples, 64*6))
        x_vs = np.zeros((x_v.shape[0], 64*6))
        layer = Model(inputs=model.input, outputs=model.get_layer(name).output)
        print "Loading ", len(x_t), " statistics for training set: layer ", name
        #for idx, im in enumerate(x_t):  # get new features and cat old ones
        x_ts = gcnn.extract_features(x_t, x_ts, layer)
        print "layer output: feature maps of dim: {}".format(x_ts.shape[-1]/6.)

        if len(x_v_real) > 0:
            n_take = len(x_ts) - len(x_v_real)
            x_ts[n_take:] = x_v_real
            y_t_ = np.concatenate((y_t, y_v_real), axis=0)
            x_ts, y_t = shuffle(x_ts, y_t_)
            print "added the old {} examples".format(len(x_v_real))

        print "Loading ", len(x_v), "statistics for valid set: layer ", name
        # for idx, im in enumerate(x_v):  # get new test features for reals
        x_vs = gcnn.extract_features(x_v, x_vs, layer)

        print "cascading:\ntrainx: {}, valx: {}".format(x_ts.shape, x_vs.shape)
        print "trainy: {}, valy: {}".format(y_t.shape, y_v.shape)
        (p_real, p_adv), (n_real, n_adv) = gsvm.cascade_svm((x_ts, y_t), (x_vs, y_v))

        old_accuracy = accuracy
        accuracy = float(len(p_real)+len(p_adv))/len(x_v)
        improvement = accuracy - old_accuracy
        real_left = len(p_real)
        # adv_left = len(p_adv)
        n_examples += real_left
        x_v_real = np.take(x_vs, p_real, axis=0)
        y_v_real = np.take(y_v, p_real, axis=0)

        # remove correctly identified items from test set
        new_valx_adv = np.take(x_v, n_adv, axis=0)
        new_valx_adv2 = np.take(x_v, p_adv, axis=0)
        new_valx_reals = np.take(x_v, n_real, axis=0)
        x_v = np.concatenate((new_valx_reals, new_valx_adv, new_valx_adv2), axis=0)

        new_valy_adv = np.take(y_v, n_adv, axis=0)
        new_valy_adv2 = np.take(y_v, p_adv, axis=0)
        new_valy_reals = np.take(y_v, n_real, axis=0)
        y_v = np.concatenate((new_valy_reals, new_valy_adv, new_valy_adv2), axis=0)

        x_v, y_v = shuffle(x_v, y_v)
        detected = (total_adv - len(n_adv)) / total_adv
        print "xval shape: ", x_v.shape, "yval shape: ", y_v.shape
        print "Round accuracy: {}".format(accuracy)
        print "Improvement: {}".format(improvement)
        print "New validation lineup: {"
        print "\tReals eliminated: {}".format(real_left)
        print "\tAdversarials eliminated this round: {}".format(len(p_adv))
        print "\tAdversarials left in pipeline: {}".format(len(n_adv))
        print "\tTotal adversarials eliminated: {} %".format(detected)
        print "}"
        print "Handing to new SVM round\n"


def self_aware_model(train, test, wpath):

    gcnn.train_self_aware_model(train, test, wpath)

def main():

    converged = 0
    it = start_iter
    lbfgs.config_tf()

    # train_base()
    wpath = wpath_base+str(it)+'.h5'
    while not converged:
        print "ITERATION ", it
        if LOG:
            logger("Iteration {}\n".format(it))

        # generate_adv(wpath, it)
        old_wpath = wpath
        wpath = old_wpath[:-5]+str(it+1)+'.h5'
        adv_dir = save_dir_base+str(it)+'/'

        train, test = load_data.load_all_resnet(adv_dir, images_dir)

        # acc, _ = ft_detector(it, train, test, wpath[:-3], old_wpath)
        self_aware_model(train, test, wpath)
        cascade_classifier(train, test)


        if acc < .1:
            converged = 1
        it += 1

    if LOG:
        logger("converged after {} iterations\n".format(it))
    print "converged after {} iterations".format(it)


if __name__ == '__main__':

    main()
