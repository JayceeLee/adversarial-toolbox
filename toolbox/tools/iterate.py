import os
import sys
sys.path.append('../')
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

home = '/home/neale/repos/adversarial-toolbox'
save_dir_base = home+'/images/adversarials/lbfgs/imagenet/test1/resnet_test'
images_dir = home+'/images/imagenet12/train/'
wpath_base = home+'/toolbox/models/weights/detectors/lbfgs/imagenet/iter_'
wpath_init = 'iter_0'
start_iter = 0
min_nonzero = 1000
log_file = './cnn_scores.txt'
LOG = True
n_images = 10000

def logger(data):

    with open(log_file, 'a+b') as f:
        f.write(data)


def generate_adv(wpath, it):

    num_gen, start = 0, 2000
    if it == 0:
        targets = 1000
        top = 'vanilla'
    else:
        targets = 2
        top = 'detector'
    (x, _), (x_val, _) = load_data.load_real(targets,
                                             images_dir,
                                             'JPEG',
                                             (224, 224, 3)
    #score = np.inf
    #while score > 10000:
    model = lbfgs.load_model(wpath, top)
        #print "scoring"
        #score = lbfgs.score_dataset(model, x, 1)
        #print "misses on real data: {}".format(score)
        #if score < 10000:
        #    print "score too low, reloading"

    for i, img in enumerate(x[start:n_images]):
        # lbfgs.display_im(img)
        adv_obj = lbfgs.generate(model, img, targets=targets)
        if adv_obj is None:
            print "Failed to apply attack"
            continue
        adv_img = adv_obj.image
        # lbfgs.display_im(adv_img)
        if min_nonzero and (np.count_nonzero(adv_img) < min_nonzero):
            print "FAILED, too many zero pixels"
            continue
        res = lbfgs.print_stats(adv_obj, model)
        if res == 1:
            print "FAILED with img {}".format(i)
            continue
        num_gen += 1
        res_dir = save_dir_base+'{}/'.format(str(it))
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        imsave(res_dir+'im_{}.png'.format(num_gen), adv_img)
        print "SUCCESS, images saved {}".format(num_gen)
        print "Images attempted: {}".format(i+1)
        print "------------------------"
        sys.exit(0)
    if LOG:
        logger('generated {} images with L-BFGS\n'.format(num_gen))

    return num_gen


def ft_detector(it, train, test, wpath, old_wpath):

    model, hist = td.improve_detector(train, test, old_wpath)
    td.test_cifar(model)
    acc = hist[-1]
    path = wpath + '.h5'  # + str(acc)[:4] + '.h5'
    model.save_weights(path)
    print "saving weights to {}".format(path)
    if LOG:
        logger('fine tuned CNN to {}% accuracy\n'.format(acc*100))


def train_base():

    model = cifar_base.train()
    model.save_weights(wpath_base+wpath_init)


def svm(train, test, it):

    clf, test_data = gs.train_svm(train, test)
    acc = gs.test_svm(clf.best_estimator_, test_data)
    joblib.dump(clf.best_estimator_, wpath_base+'{}_svm_{}.pkl'.format(it, acc))
    if LOG:
        logger('svm iteration {}\n'.format(it))
        logger('accuracy: {}\n'.format(acc))
        logger('GridCV results: \n')
        for k, v in sorted(clf.best_params_.items()):
            logger(str(k) + ' ' + str(v) + '\n')
        logger('\n\n------------------------------------\n')
    return acc


def grid_svm():

    for i in range(22):
        logger('testing svm {} on all datasets'.format(i))
        svm_str = wpath_base+'{}_svm.pkl'.format(i)
        print "loading SVM {} from {}".format(i, svm_str)
        clf = joblib.load(svm_str)
        for j in range(22):
            adv_dir = save_dir_base+'{}/'.format(j)
            print "loading adversarial images from {}".format(adv_dir)
            _, (x, y) = load_data.load_real_adv(adv_dir, (32, 32))

            x_grad = gs.collect_gradients(x, 32)
            dim = 32 * 32
            x_grad = x_grad.reshape(x_grad.shape[0], dim)
            y = np.argmax(y, axis=1)

            acc = gs.test_svm(clf, (x_grad, y))
            print "SVM {} on data {}: {}%".format(i, j, acc)
            logger(str(acc)+'\n')


def main():

    converged = 0
    it = start_iter
    lbfgs.config_tf()

    # train_base()
    # grid_svm()
    wpath = wpath_base+str(it)+'.h5'
    while not converged:
        print "ITERATION ", it
        if LOG:
            logger("Iteration {}\n".format(it))

        generate_adv(wpath, it)

        old_wpath = wpath
        wpath = old_wpath[:-5]+str(it+1)+'.h5'
        adv_dir = save_dir_base+str(it)+'/'

        train, test = load_data.load_real_adv(adv_dir, (32, 32))
        ft_detector(it, train, test, wpath[:-3], old_wpath)

        acc = svm(train, test, it)
        if acc < .1:
            converged = 1
        it += 1

    if LOG:
        logger("converged after {} iterations\n".format(it))
    print "converged after {} iterations".format(it)


if __name__ == '__main__':

    main()
