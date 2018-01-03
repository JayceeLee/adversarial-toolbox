import os
import sys
import numpy as np
# import argparse
from imageio import imwrite
from scipy.misc import imread
from keras.applications.resnet50 import preprocess_input

from gen_tools import validate_label, load_sorted, deprocess
from gen_tools import check_failure, print_preds, plot_distortion
from gen_tools import load_resnet_type

sys.path.append('../')
from attacks import mifgsm_attack, foolbox_attack

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

home = '/home/neale/repos/adversarial-toolbox'
save_dir_adv = home+'/images/adversarials/cw/imagenet/symmetric/adv'
save_dir_real = home+'/images/adversarials/cw/imagenet/symmetric/real'
images_dir = home+'/images/imagenet12/fcn_train/'
real_diff_dir = home+'/images/adversarials/cw/imagenet/symmetric/avg_real/'
adv_diff_dir = home+'/images/adversarials/cw/imagenet/symmetric/avg_adv/'
diff_dir = home+'/images/adversarials/cw/imagenet/symmetric/diff/'

wpath_base = home+'/toolbox/models/weights/detectors/deepfool/imagenet/iter_'
wpath_init = 'iter0'
start_iter = 0
min_nonzero = 1000
n_images = 1000

ilsvrc_x = home+'/images/imagenet12/val/'
ilsvrc_y = home+'/images/val_num.txt'


def load_args():
    pass


def generate(wpath, it):

    num_gen = 0
    start = 0
    model = load_resnet_type(vanilla=True)
    model2 = load_resnet_type(vanilla=True)
    train = load_sorted(100)
    img_set = train
    idx = start

    for i, x in enumerate(img_set):
        real = np.copy(x)
        x = x.astype(np.float32)
        fx = preprocess_input(np.expand_dims(x, 0))
        print "** original labels **"
        print_preds(model, fx)
        idx += 1
        label = np.argmax(model.predict(fx))
        fx_adv = mifgsm_attack.mifgsm(model, fx, label, 1000)[0]

        x_adv = deprocess(fx_adv).astype(np.uint8)
        res = check_failure(fx_adv, model)
        if res[0] > 0:
            print res[1]
            continue
        # plot_distortion(real, fx_adv)
        x = np.copy(x_adv)
        print "** adversarial labels **"
        print_preds(model, fx_adv)
        diff = real - x
        print "adversarial norm: ", np.linalg.norm(diff)
        if validate_label(model, real) == validate_label(model, x):
            print "failed with img: {}".format(idx)
            print "-----------------------------------------"
            continue

        for d in [save_dir_real, save_dir_adv, diff_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        save_str_real = save_dir_real+'/im_{}.png'.format(idx)
        save_str_adv = save_dir_adv+'/adv_{}.png'.format(idx)
        save_str_diff = diff_dir+'/diff_{}.png'.format(idx)
        print "saving to ", save_str_adv

        imwrite(save_str_real, real)
        imwrite(save_str_adv, x_adv)
        imwrite(save_str_diff, diff)

        valid = imread(save_str_adv)
        pvalid = preprocess_input(np.expand_dims(valid, 0))
        print "loaded ",  print_preds(model, pvalid, name=True)
        if (validate_label(model, valid, name=False) ==
                validate_label(model, real, name=False)):
            print "failed: did not survive quantization"
            os.remove(save_str_adv)
            os.remove(save_str_real)
            os.remove(save_str_diff)
            print "-----------------------------------------"
            continue

        num_gen += 1
        print "success, images saved {}".format(num_gen)
        print "images attempted for this run: {}".format(i+1)
        print "-----------------------------------------"

    return num_gen


def main():

    it = start_iter
    foolbox_attack.config_tf(gpu=.8)
    wpath = wpath_base+str(it)+'.h5'
    generate(wpath, it)


if __name__ == '__main__':

    main()
