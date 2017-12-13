import os
import sys
import load_data
sys.path.append('../')

import numpy as np
from glob import glob
from attacks import lbfgs
from imageio import imwrite
from models import cifar_model
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.misc import imsave, imread

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import SGD
from keras.layers import AveragePooling2D, Input
from keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"]="1"

home = '/home/neale/repos/adversarial-toolbox'
save_dir_adv = home+'/images/adversarials/deepfool/imagenet/symmetric/adv'
save_dir_real = home+'/images/adversarials/deepfool/imagenet/symmetric/real'
images_dir = home+'/images/imagenet12/fcn_train/'
real_diff_dir = home+'/images/adversarials/deepfool/imagenet/symmetric/avg_real/'
adv_diff_dir = home+'/images/adversarials/deepfool/imagenet/symmetric/avg_adv/'
diff_dir = home+'/images/adversarials/deepool/imagenet/symmetric/diff/'
wpath_base = home+'/toolbox/models/weights/detectors/deepfool/imagenet/iter_'
wpath_init = 'iter0'
start_iter = 0
min_nonzero = 1000
n_images = 1000

ilsvrc_x = home+'/images/imagenet12/val/'
ilsvrc_y = home+'/images/val_num.txt'


def validate_label(model, x, name=False):
    if x.ndim < 3:
        raise ValueError("Need at least a three dimensional input")
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    x = preprocess_input(x.astype(np.float32))
    preds = model.predict(x)
    if name:
        pred = decode_predictions(preds, top=2)
    else:
        if x.ndim == 4:
            pred = np.argmax(preds, axis=1)
        else:
            pred = np.argmax(preds)
    return pred


def load_sorted(n_images, im_dir=images_dir, suff='JPEG'):

    paths = glob(im_dir+'*.'+suff)
    paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths = paths[:n_images]
    x = np.empty((len(paths), 224, 224, 3), dtype=np.uint8)
    x = load_data.load_dir(paths, arr=x, start=0, end=len(paths))

    return x


def load_resnet_type(vanilla=True, pool=False, bilateral=False):

    args = [vanilla, pool, bilateral]
    assert sum(args) == 1, "Can only define one model at a time"
    if vanilla is True:
        model = ResNet50(weights='imagenet', include_top=True)
    elif pool is True:
        dim = (224, 224, 3)
        inputs = Input(shape=dim)
        x = (AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
             (inputs))
        model = ResNet50(weights='imagenet', include_top=True, input_tensor=x)
    elif bilateral is True:
        raise NotImplementedError
    return model


def load_irv2_type(vanilla=True, pool=False, bilateral=False):

    args = [vanilla, pool, bilateral]
    assert sum(args) == 1, "Can only define one model at a time"
    if vanilla is True:
        model = InceptionResNetV2(weights='imagenet', include_top=True)
    elif pool is True:
        dim = (224, 224, 3)
        inputs = Input(shape=dim)
        x = (AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
             (inputs))
        model = InceptionResNetV2(weights='imagenet', include_top=True, input_tensor=x)
    elif bilateral is True:
        raise NotImplementedError
    return model


def train_val(save_name):
    images, labels = load_data.load_ilsvrc_labeled(24000, ilsvrc_x, ilsvrc_y)
    model = load_resnet_type(vanilla=True)
    labels = to_categorical(labels, 1000)
    val_x = images[:2000]
    train_x = images[2000:]
    val_y = labels[:2000]
    train_y = labels[2000:]

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[:2]:
        layer.trainable = True
    model.summary()

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(train_x,
              train_y,
              epochs=10,
              batch_size=32,
              shuffle=True,
              validation_data=(val_x, val_y))

    model.save_weights(save_name+'.h5')


def check_failure(adv, model):

    ret = (0, "success")

    if type(adv) is None:
        ret = (1, "adversarial sent back is missing") 

    elif min_nonzero and (np.count_nonzero(adv) < min_nonzero):
        ret = (2, "zeroed adversarial, bad network input")

    return ret


def diff_images():

    print "loading real images"
    real = load_sorted(10000, real_diff_dir, suff='png')
    print "loading adversarial images"
    adv = load_sorted(10000, adv_diff_dir, 'png')

    real = preprocess_input(real)
    adv = preprocess_input(adv)
    for idx, (r, a) in enumerate(zip(real, adv)):
        diff = r - a
        imsave(diff_dir+'{}.png'.format(idx), diff)


def print_preds(kmodel, fmodel, kx, fx):

        fpred = np.argmax(fmodel.predictions(fx))
        klabel = validate_label(kmodel, kx, name=True)[0][0]
        kpred = validate_label(kmodel, kx, name=False)[0]

        print "Foolbox: ", fpred
        print "Keras  : ", kpred, klabel


def generate_adv(wpath, it):

    num_gen = 0
    start = 0
    targets = 1000
    
    net = 'resnet'
    print "loading from ", wpath
    kmodel = load_resnet_type(vanilla=True)
    # kmodel = load_irv2_type(vanilla=True)
    # kmodel.load_weights('./avg_resnet50.h5')
    model = lbfgs.load_fmodel(model=kmodel, net='resnet')
    kmodel.summary()

    val_model = ResNet50(weights='imagenet')
    # val_model = InceptionResNetV2(weights='imagenet')
    train = load_sorted(10000)
    img_set = train[start:]
    idx = start

    for i, x in enumerate(img_set):
        real = np.copy(x) # copy of image (keep)
        x = x.astype(np.float32)
        
        if net == 'ir':
            fx = x / 127.5
            fx = fx - 1
        elif net == 'resnet':
            fx = x[..., ::-1] # (foolbox) invert image before pp
        
        print "** Original Labels **"
        print_preds(val_model, model, x, fx)
        print "Performing L-BFGS-B attack"
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
            x_adv = x_adv[..., ::-1] # (foolbox) invert back to normal

        print "** Adversarial Labels **"
        print_preds(val_model, model, x_adv, fx_adv)
        print "Adversarial Norm: ", np.linalg.norm(real-x_adv)
        if validate_label(val_model, real) == validate_label(val_model, x_adv):
            print "FAILED with img: {}".format(idx)
            continue
        
        num_gen += 1
        for d in [save_dir_real, save_dir_adv, diff_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        diff = real - x_adv
        save_str_real = save_dir_real+'/im_{}.png'.format(idx)
        save_str_adv = save_dir_adv+'/adv_{}.png'.format(idx)
        save_str_diff = diff_dir+'/diff_{}.png'.format(idx)
        print "saving to ", save_str_adv

        imsave(save_str_real, real)
        imsave(save_str_adv, x_adv)
        imsave(save_str_diff, diff)
        
        valid = imread(save_str_adv)
        print "loaded ",  validate_label(val_model, valid, name=True)
        
        print "SUCCESS, images saved {}".format(num_gen)
        print "Images attempted for this run: {}".format(i+1)
        print "-----------------------------------------"

    return num_gen


def main():

    it = start_iter
    lbfgs.config_tf(gpu=.8)
    wpath = wpath_base+str(it)+'.h5'
    # diff_images()
    generate_adv(wpath, it)


if __name__ == '__main__':

    main()
