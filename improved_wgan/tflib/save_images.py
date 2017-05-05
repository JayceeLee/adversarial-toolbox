"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""
import os
import numpy as np
import scipy.misc
from scipy.misc import imsave
import matplotlib.pyplot as plt

def save_images(X, save_path, individual=False):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))

    result_dir = os.getcwd()+'/samples/cifar10/composite/pool2_no_inter_pool/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if individual == True:
        for i, im in enumerate(X):
            im = np.reshape(im, (32, 32, 3)).astype(np.uint8)
            imsave(result_dir+save_path+'_'+str(i)+'.png', im)

    else:

        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples/rows

        if X.ndim == 2:
            X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

        if X.ndim == 4:
            # BCHW -> BHWC
            X = X.transpose(0,2,3,1)
            h, w = X[0].shape[:2]
            img = np.zeros((h*nh, w*nw, 3))

        elif X.ndim == 3:
            h, w = X[0].shape[:2]
            img = np.zeros((h*nh, w*nw))


        for n, x in enumerate(X):

            j = n/nw
            i = n%nw
            img[j*h:j*h+h, i*w:i*w+w] = x

        imsave(result_dir+save_path+'.png', img)
