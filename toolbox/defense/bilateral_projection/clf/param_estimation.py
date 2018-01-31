import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import metrics
import numpy as np
from sklearn import decomposition
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.decomposition import PCA
import tensorflow as tf
import cv2
import tf_models
from bilateral_tf_mim import load_images

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


def load_params():
    x = np.load('./resnet_v2_mim_500_images.npy')
    print x.shape
    y = np.load('./resnet_v2_mim_500_params.npy')
    return (x, y)


def collect_gradients(data, arr, c=1):

    # Using three channels seems to be too sparse, one channel works
    # But we're going to give it more of and effort with large images
    if c == 1:
        for i in range(len(data)):
            im = data[i]#.astype(np.int32)
            gradients = np.zeros((len(data), 299, 299))
            r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            imx = np.zeros((299, 299))
            ndimage.sobel(gray, 1, imx)
            imy = np.zeros((299, 299))
            ndimage.sobel(gray, 0, imy)
            mag = np.sqrt(imx**2+imy**2)
            gradients[i] = mag
        return gradients
    else:
        for i in range(len(data)):
            im = data[i].astype(np.int32)
            gradients = np.zeros(im.shape)
            channels = im.shape[2]
            for j in range(channels):
                imx = np.zeros((299, 299))
                ndimage.sobel(im[:, :, j], 1, imx)
                imy = np.zeros((299, 299))
                ndimage.sobel(im[:, :, j], 0, imy)
                mag = np.sqrt(imx**2+imy**2)
                gradients[:, :, j] = mag

        arr[i] = gradients

    print "\n==> gradient data shape\n", arr.shape

    return arr


images, labels = load_params()

print images.shape
print labels.shape
n_samples = len(images)

arr = np.zeros((len(images), 299, 299, 3))

x_grad = collect_gradients(images, arr)

# Split the data into training/testing sets
"""
flat_x = np.zeros((len(images), 299*299))
for i in range(len(x_grad)):
    flat_x[i] = x_grad[i].flatten()
"""
flat_x = np.zeros(len(images))
for i in range(len(x_grad)):
    flat_x[i] = np.linalg.norm(x_grad[i])

x_train = flat_x[:len(flat_x)400]
x_test = flat_x[400:]
y_train = labels[:400]
y_test = labels[400:]

"""
z = []
for i in range(len(labels)):
    print labels[i]
    if labels[i][0] != 0.:
        z.append(i)
nonzero_x = images[z]
nonzero_y = labels[z]
assert len(nonzero_x) == len(nonzero_y)

x_train = nonzero_x
y_train = nonzero_y
print x_train.shape
print y_train.shape
npoints = 10000
fshape = 3
feature_set = np.zeros((len(x_train), 18)) 
for k, x in enumerate(x_train):
    K = x.shape[-1]
    samples = np.zeros((npoints, K))
    rows = np.random.randint(x.shape[0], size=npoints) # we want to draw n pixels
    cols = np.random.randint(x.shape[1], size=npoints) # we want to draw n pixels
    for i, (r, c) in enumerate(zip(rows, cols)):
        samples[i] = x[r][c]
    means = np.mean(samples, axis=0) # gives K means
    stds = np.std(samples, axis=0)
    pca = PCA()
    samplesT = np.transpose(samples)
    statistics = np.zeros(fshape)
    for i, p in enumerate(samplesT):
        dim = int(np.sqrt(p.shape[0]))
        p = np.reshape(p, (dim, dim))
        pca.fit(p)
        coeffs = pca.transform(p)
        proj = np.transpose(coeffs) * (p - means[i])
        projZ = proj/(stds[i])
        x = 1./npoints * np.linalg.norm(projZ, ord=2)
        statistics[i] = x

    max_samples = np.max(samples, axis=0)
    min_samples = np.min(samples, axis=0)
    samples25 = np.zeros(fshape)
    samples50 = np.zeros(fshape)
    samples75 = np.zeros(fshape)
    for i, p in enumerate(samplesT):
        psort = sorted(p)[::-1]
        samples25[i] = psort[len(p)//4]
        samples50[i] = psort[len(p)//2]
        samples75[i] = psort[3*(len(p)//4)]

    features = np.concatenate((statistics,
        max_samples,
        min_samples,
        samples25,
        samples50,
        samples75),
        axis=0)
    feature_set[k] = features
"""
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train[:-50], y_train[:-50])

# Make predictions using the testing set
y_pred = regr.predict(x_train[-50:])

for i in range(len(y_pred)):
    y_pred[i] = map(int, y_pred[i])
    print y_pred[i]


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
        % mean_squared_error(y_train[-50:], y_pred))

print "score: {}".format(regr.score(x_train[-50:], y_train[-50:]))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_train[-50:], y_pred))

import sys
sess = tf.Session()
model = tf_models.InceptionV3Model(sess=sess)
model._build()
hit = 0 
for i, pred in enumerate(y_pred):
    y = np.argmax(model.predict(np.expand_dims(rimages[400+i].astype(np.float32), 0)))
    x = np.argmax(model.predict(np.expand_dims(images[400+i].astype(np.float32), 0)))
    if x != y:
        k = int(pred[0])
        sc = int(pred[1])
        ss = int(pred[2])
        print k, sc, ss
        try:
            z = cv2.bilateralFilter(images[400+i].astype(np.float32), k, sc, ss)
        except Exception as e:
            print "Unexpected error:", e
            continue
        zpred = np.argmax(model.predict(np.expand_dims(z, 0)))
        if zpred == y:
            print "success"
            hit += 1

print "hits {}%".format(float(hit)/len(y_pred))
