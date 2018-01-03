import os
import sys
import numpy as np
import argparse
from imageio import imwrite
# import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input

from gen_tools import validate_label, load_sorted
from gen_tools import check_failure, print_preds
from gen_tools import load_resnet_type

sys.path.append('../')
from attacks import lbfgs, cw_attack
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


def generate_adv_tf(wpath, it):

    num_gen = 0
    start = 0
    net = 'resnet'
    print "loading from ", wpath
    model = load_resnet_type(vanilla=True)
    val_model = load_resnet_type(vanilla=True)
    # kmodel = load_irv2_type(vanilla=true)
    # val_model = inceptionresnetv2(weights='imagenet')
    train = load_sorted(100)
    img_set = train
    idx = start

    for i, x in enumerate(img_set):
        real = np.copy(x)
        x = x.astype(np.float32)

        if net == 'ir':
            fx = x / 127.5
            fx = fx - 1
        elif net == 'resnet':
            fx = preprocess_input(np.expand_dims(x, 0))  # (foolbox) invert image before pp
        print "** original labels **"
        idx += 1
        label = np.argmax(model.predict(fx))
        print label
        fx_adv = cw_attack.cw_attack(model, fx[0], label, norm='2')
        res = check_failure(fx_adv, val_model)
        if res[0] > 0:
            print res[1]
            continue

        x_adv = np.copy(fx_adv)
        if net == 'ir':
            x_adv = x_adv + 1
            x_adv = x_adv * 127.5
        else:
            x_adv *= 255
            print x_adv.shape
            x_adv = np.reshape(x_adv[0], (224, 224, 3))
            # x_adv = x_adv[..., ::-1]  # (foolbox) invert back to normal


        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x)
        plt.subplot(1, 3, 2)
        plt.imshow(x_adv)
        plt.subplot(1, 3, 3)
        plt.imshow(x-x_adv)
        plt.show()
        print "** adversarial labels **"
        print_preds(val_model, model, x_adv, fx_adv)
        print "adversarial norm: ", np.linalg.norm(real-x_adv)
        if validate_label(val_model, real) == validate_label(val_model, x_adv):
            print "failed with img: {}".format(idx)
            print "-----------------------------------------"
            continue

        for d in [save_dir_real, save_dir_adv, diff_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        diff = real - x_adv
        save_str_real = save_dir_real+'/im_{}.png'.format(idx)
        save_str_adv = save_dir_adv+'/adv_{}.png'.format(idx)
        save_str_diff = diff_dir+'/diff_{}.png'.format(idx)
        print "saving to ", save_str_adv

        imwrite(save_str_real, real)
        imwrite(save_str_adv, x_adv)
        imwrite(save_str_diff, diff)

        valid = imwrite(save_str_adv)
        print "loaded ",  validate_label(val_model, valid, name=True)
        if (validate_label(val_model, valid, name=False) ==
                validate_label(val_model, real, name=False)):
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


def generate_adv_keras(wpath, it):

    num_gen = 19999
    start = 19999
    targets = 1000

    net = 'resnet'
    print "loading from ", wpath
    kmodel = load_resnet_type(vanilla=True)
    # kmodel = load_irv2_type(vanilla=true)
    # kmodel.load_weights('./avg_resnet50.h5')
    model = lbfgs.load_fmodel(model=kmodel, net='resnet')
    kmodel.summary()

    val_model = load_resnet_type(vanilla=True)
    # val_model = inceptionresnetv2(weights='imagenet')
    train = load_sorted(20000)
    img_set = train
    idx = start

    for i, x in enumerate(img_set):
        real = np.copy(x)
        x = x.astype(np.float32)

        if net == 'ir':
            fx = x / 127.5
            fx = fx - 1
        elif net == 'resnet':
            fx = x[..., ::-1]  # (foolbox) invert image before pp

        print "** original labels **"
        print_preds(val_model, model, x, fx)
        print "performing l-bfgs-b attack"
        idx += 1

        fx_adv = lbfgs.generate(model, fx, targets=targets)
        res = check_failure(fx_adv, val_model)
        if res[0] > 0:
            print res[1]
            continue

        x_adv = np.copy(fx_adv)
        if net == 'ir':
            x_adv = x_adv + 1
            x_adv = x_adv * 127.5
        else:
            x_adv = x_adv[..., ::-1]  # (foolbox) invert back to normal

        print "** adversarial labels **"
        print_preds(val_model, model, x_adv, fx_adv)
        print "adversarial norm: ", np.linalg.norm(real-x_adv)
        if validate_label(val_model, real) == validate_label(val_model, x_adv):
            print "failed with img: {}".format(idx)
            print "-----------------------------------------"
            continue

        for d in [save_dir_real, save_dir_adv, diff_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        diff = real - x_adv
        save_str_real = save_dir_real+'/im_{}.png'.format(idx)
        save_str_adv = save_dir_adv+'/adv_{}.png'.format(idx)
        save_str_diff = diff_dir+'/diff_{}.png'.format(idx)
        print "saving to ", save_str_adv

        imwrite(save_str_real, real)
        imwrite(save_str_adv, x_adv)
        imwrite(save_str_diff, diff)

        valid = imwrite(save_str_adv)
        print "loaded ",  validate_label(val_model, valid, name=True)
        if (validate_label(val_model, valid, name=False) ==
                validate_label(val_model, real, name=False)):
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
    lbfgs.config_tf(gpu=.8)
    wpath = wpath_base+str(it)+'.h5'
    # diff_images()
    generate_adv_tf(wpath, it)


if __name__ == '__main__':

    main()
