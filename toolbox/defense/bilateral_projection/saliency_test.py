# Boilerplate imports.
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
slim=tf.contrib.slim

old_cwd = os.getcwd()
os.chdir('models/research/slim')
from nets import inception_v3
os.chdir(old_cwd)

# From our repository.
import saliency
# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    im = ((im + 1) * 127.5).astype(np.uint8)
    P.imshow(im)
    P.title(title)
    P.show()

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)
    P.show()

def ShowDivergingImage(grad, title='', percentile=99, ax=None):  
    if ax is None:
        fig, ax = P.subplots()
    else:
      fig = ax.figure

    P.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    return im / 127.5 - 1.0

x = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/adv_npy/adv_4.png.npy')
y = np.load('../../../images/adversarials/mim/imagenet/symmetric/inception_v3/real_npy/real_4.png.npy')

im1 = x
im2 = y

ckpt_file = './inception_v3.ckpt'
graph = tf.Graph()
with graph.as_default():
    images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        _, end_points = inception_v3.inception_v3(images, is_training=False, num_classes=1001)

# Restore the checkpoint
    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

# Construct the scalar neuron tensor.
    logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]

# Construct tensor for predictions.
    prediction = tf.argmax(logits, 1)

    ShowImage(im2)

# Make a prediction. 
    prediction_class = sess.run(prediction, feed_dict = {images: [im1]})[0]

    print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 237
    """
    gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)

# Compute the vanilla mask and the smoothed mask.
    vanilla_mask_3d = gradient_saliency.GetMask(im, feed_dict = {neuron_selector: prediction_class})
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, feed_dict = {neuron_selector: prediction_class})

# Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

# Set up matplot lib figures.
    ROWS = 1
    COLS = 2
    UPSCALE_FACTOR = 10
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
    ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient', ax=P.subplot(ROWS, COLS, 1))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))
    """
    guided_backprop = saliency.GuidedBackprop(graph, sess, y, images)

# Compute the vanilla mask and the smoothed mask.
    for im in [im1, im2]:
        vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(
                im, feed_dict = {neuron_selector: prediction_class})
        smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(
                im, feed_dict = {neuron_selector: prediction_class})

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_guided_backprop_mask_3d)
        smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop_mask_3d)

# Set up matplot lib figures.
        ROWS = 1
        COLS = 2
        UPSCALE_FACTOR = 10
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Render the saliency masks.
        ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Guided Backprop', ax=P.subplot(ROWS, COLS, 1))
        ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad Guided Backprop', ax=P.subplot(ROWS, COLS, 2))
