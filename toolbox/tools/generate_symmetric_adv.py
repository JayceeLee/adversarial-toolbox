import os
import sys
import keras
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import train_detector as td
import gcnn
import gsvm as gs
from attacks import lbfgs
from scipy.misc import imsave, imread
from glob import glob
from models import cifar_model
import load_data
from sklearn.externals import joblib
import cifar_base
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

home = '/home/neale/repos/adversarial-toolbox'
save_dir_base = home+'/images/adversarials/deepfool/imagenet/symmetric/resnet_test'
images_dir = home+'/images/imagenet12/recovery_set_all/'
wpath_base = home+'/toolbox/models/weights/detectors/deepfool/imagenet/iter_'
wpath_init = 'iter0'
start_iter = 0
min_nonzero = 1000
log_file = './recovery_scores.txt'
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


def load_sorted_real():

    paths = glob(images_dir+'*.JPEG')
    paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    x = np.empty((len(paths), 224, 224, 3))
    x = load_data.load_dir(paths, arr=x, start=0, end=len(paths))
    return x


def generate_adv(wpath, it):

    num_gen = 0
    start = 0
    targets = 1000

    train = load_sorted_real()
    print "loading from ", wpath
    kmodel = ResNet50(weights='imagenet')
    model = lbfgs.load_model(model=kmodel)
    kmodel.summary()
    val_model = ResNet50(weights='imagenet')
    img_set = train[start:n_images]
    idx = 0
    for i, img in enumerate(img_set):
        idx += 1
        # lbfgs.display_im(img)
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
        save_str = res_dir+'{}.png'.format(idx)
        print "saving to ", save_str

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


def main():

    it = start_iter
    lbfgs.config_tf()
    wpath = wpath_base+str(it)+'.h5'
    generate_adv(wpath, it)


if __name__ == '__main__':

    main()
