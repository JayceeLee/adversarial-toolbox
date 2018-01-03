import load_data
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import AveragePooling2D, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

home = '/home/neale/repos/adversarial-toolbox'
save_dir_adv = home+'/images/adversarials/deepfool/imagenet/symmetric/adv'
save_dir_real = home+'/images/adversarials/deepfool/imagenet/symmetric/real'
images_dir = home+'/images/imagenet12/fcn_train/'
real_diff_dir = home+'/images/adversarials/deepfool/imagenet/symmetric/avg_real/'
adv_diff_dir = home+'/images/adversarials/deepfool/imagenet/symmetric/avg_adv/'
diff_dir = home+'/images/adversarials/deepfool/imagenet/symmetric/diff/'
wpath_base = home+'/toolbox/models/weights/detectors/deepfool/imagenet/iter_'

wpath_init = 'iter0'
start_iter = 0
min_nonzero = 1000
n_images = 1000

ilsvrc_x = home+'/images/imagenet12/val/'
ilsvrc_y = home+'/images/val_num.txt'


def deprocess(x):

    if x.ndim == 3:
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[..., ::-1]

    elif x.ndim == 4:
        x[:, :, :, 0] += 103.939
        x[:, :, :, 1] += 116.779
        x[:, :, :, 2] += 123.68
        x = x[..., ::-1]
    else:
        raise AttributeError("Image must be singular, or a batch of images")
    return x


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


def load_sorted(n_images, im_dir=images_dir, size=224, suff='JPEG'):

    paths = glob(im_dir+'*.'+suff)
    paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    paths = paths[:n_images]
    x = np.empty((len(paths), size, size, 3), dtype=np.uint8)
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
    diffs = np.empty((len(real), 224, 224, 3))
    for idx, (r, a) in enumerate(zip(real, adv)):
        diff = r - a
        diffs[idx] = diff
    return diffs


def plot_distortion(x, x_adv):

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(x)
        plt.subplot(1, 3, 2)
        plt.imshow(x_adv)
        plt.subplot(1, 3, 3)
        plt.imshow(x-x_adv)
        plt.show()


def print_preds(kmodel, kx):

        vlabel = validate_label(kmodel, kx, name=True)[0][0]
        vpred = validate_label(kmodel, kx, name=False)[0]

        print "Keras  : ", vpred, vlabel


def print_preds_fb(kmodel, fmodel, kx, fx):

        fpred = np.argmax(fmodel.predictions(fx))
        klabel = validate_label(kmodel, kx, name=True)[0][0]
        kpred = validate_label(kmodel, kx, name=False)[0]

        print "Foolbox: ", fpred
        print "Keras  : ", kpred, klabel
