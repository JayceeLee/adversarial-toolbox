import os, sys
sys.path.append(os.getcwd())

import time
#import glob

import numpy as np
import tensorflow as tf
import matplotlib
import functools
matplotlib.use('pdf')
from skimage import color
from scipy.ndimage import filters
import sklearn.datasets
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.small_imagenet
import tflib.cub128
import tflib.cifar10
import tflib.ops.layernorm
import tflib.save_images
import tflib.inception_score
import tflib.plot
from keras.models import Sequential
#import tflib.inception_score
#import tflib.plot
import keras.backend as K
#from keras.utils import np_utils
from keras.layers.core import Flatten, Dense, Dropout
from sklearn.utils import shuffle
from genericnet import generic
import argparse
DATA_DIR = '/home/neale/repos/adversarial-toolbox/images/cub128'

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
GEN_ITERS = 1 # how many generator iterations per disc iteration (old DCGAN hack)
BATCH_SIZE = 4 # Batch size
BATCH_SIZE_ADV = 4 # Batch size
ITERS = 300000 # How many generator iterations to train for
OUTPUT_DIM = 49152 # Number of pixels in CIFAR10 (3*32*32)
D2_START_ITERS = 300000
SAVE_ITERS = [1, 10000, 20000, 50000, 100000]
SAVE_NAME = "128_wgan-gp"
lib.print_model_settings(locals().copy())

def load_args():

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-s', '--save_dir', default='default_dir',type=str, help='experiment_save_dir')
  parser.add_argument('-z', '--z', default=128,type=int, help='noise sample width')
  parser.add_argument('-t', '--tf_trial_name', default=0,type=str, help='tensorboard trial')
  args = parser.parse_args()
  return args

args = load_args()

def clf(outputs, model=False):

    if model == True:
	top_model = Sequential()
	top_model.add(Flatten(input_shape=(32, 32, 3)))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(2, activation='softmax'))
    else:
	top_model = [
	        Flatten(),
		Dense(256, activation='relu'),
		Dropout(0.5),
		Dense(1)
	]

    return top_model

def InitD2():

    detect = generic(top=False, pool=0)

    classifier = clf(1, model=False)
    for layer in classifier:
         detect.add(layer)

    detect.load_weights('/home/neale/repos/adversarial-toolbox/gp-wgan/sig_jsma_0.h5', by_name=True)
    for layer in detect.layers:
        layer.trainable=False

    K.set_learning_phase(0)
    return detect

detector = InitD2()

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def MaxPool(input, size, stride, padding):
    output = tf.nn.max_pool(input, size, stride, padding, data_format='NCHW')
    return output
def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output
def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)
base_d = False

def G(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*16*dim, noise)
    output = tf.reshape(output, [-1, 16*dim, 4, 4])

    for i in xrange(6):
        output = ResidualBlock('Generator.4x4_{}'.format(i), 16*dim, 16*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 16*dim, 8*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.8x8_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 8*dim, 4*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.16x16_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up3', 4*dim, 2*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.32x32_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up4', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(5):
        output = ResidualBlock('Generator.64x64_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up5', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = ResidualBlock('Generator.128x128_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, OUTPUT_DIM])


# ! Discriminators

def D1(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 128, 128])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])

def D2(inputs):
    """
    we need a second descriminator to fool. This one mirrors a binary
    adversarial detector. Given some sample I = G(x), we need to fool the first
    discriminator into thinking that I ~ P(x), and we also need to fool the binary
    detector into thinking that this new adversarial image comes from the natural manifold
    """
    inputs = tf.reshape(inputs, [-1, 128, 128, 3])
    d2 = detector(inputs)
    detector.summary()
    d2 = tf.reshape(d2, [-1])
    return d2

# calculate image gradient magnitudes and plot histogram

""" Here we grab samples from G and push them through D1 and D2 """
real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE*2, 3, 128, 128])
real_data_reshape = tf.reshape(real_data_int, [BATCH_SIZE*2, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_reshape, tf.float32)/255.)-.5)
gen_data = G(BATCH_SIZE*2)
train_d2 = tf.placeholder(tf.int32)
test_summ = tf.placeholder(tf.int32)
"""
We need to make a decision regarding two distributions,
We constrain our manifold to only include images from the union on sets P1 and P2
P1 being the set of natural images that could come from Cifar
P2 being the set of adversarial images that can fool our detector
"""
#disc_nat  = D2(real_data)
#disc_adv  = D2(gen_data)
disc_adv  = 1.

disc_real = D1(tf.identity(real_data))
disc_fake = D1(gen_data)

# get image gradient cdfs for generator, and real images
"""
# Print statements for the loss
disc_real = tf.Print(disc_real, [disc_real], "REAL: ")
disc_fake = tf.Print(disc_fake, [disc_fake], "FAKE: ")
disc_nat = tf.Print(disc_nat, [disc_nat], "NAT: ")
disc_adv = tf.Print(disc_adv, [disc_adv], "ADV: ")
"""
g_params = lib.params_with_name('Generator')
d1_params = lib.params_with_name('Discriminator')
#d2_params = lib.params_with_name('D2')

# Standard WGAN loss on G and first discriminator
# Control flow on the cost function. If were not using D2 we only need to consider one discriminator in our loss
#gen_cost = tf.reduce_mean(disc_adv)

gen_cost = tf.cond(train_d2 > 0, lambda: -tf.reduce_mean(disc_fake)-tf.reduce_mean(disc_adv),
                                 lambda: -tf.reduce_mean(disc_fake))

d1_cost  =  tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

""" Gradient penalty """

# TODO Verify that the penalty on the gradient doesn't apply to D2

alpha = tf.random_uniform(
    shape=[BATCH_SIZE*2,1],
    minval=0.,
    maxval=1.
)
# calculate two sided loss term, mean( sqrt( sum ( gradx^2 ) ) )
print gen_data.shape
differences = gen_data - real_data
interpolates = real_data + (alpha*differences)

gradients = tf.gradients(D1(interpolates), [interpolates])[0]
tf.summary.histogram("interpolated gradients", gradients)
grads = tf.gradients(gen_cost, tf.trainable_variables())
grads = list(zip(grads, tf.trainable_variables()))

slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
d1_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                                      gen_cost, var_list=g_params, colocate_gradients_with_ops=True
                                     )
d1_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                                       d1_cost, var_list=d1_params, colocate_gradients_with_ops=True
                                     )

# we dont want to train the detector, just load weights
"""d2_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                                       d2_cost, colocate_gradients_with_ops=True#, var_list=d2_params
                                    )
"""
d1_summ  = tf.summary.scalar("d1 cost", d1_cost)
#d2_summ  = tf.summary.scalar("d2 cost", d2_cost)
gen_summ = tf.summary.scalar("gen cost", gen_cost)

fixed_noise_z = tf.constant(np.random.normal(size=(args.z, args.z)).astype('float32'))
fixed_noise_samples_z = G(args.z, noise=fixed_noise_z)

def generate_random_image(frame,  save, board=False):
    random_noise_z = tf.constant(np.random.normal(size=(args.z, args.z)).astype('float32'))
    random_noise_samples_z = G(args.z, noise=random_noise_z)
    samples = session.run(random_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((args.z, 3, 128, 128)), 'samples_{}'.format(frame), save, individual=False)


def generate_image(frame, true_dist, save, board=False):
    samples = session.run(fixed_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((args.z, 3, 128, 128)), 'samples_{}'.format(frame), args.save_dir, individual=False)

def generate_image_tf(frame, true_dist, save):
    samples = session.run(fixed_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    return lib.save_images.save_images(samples.reshape((args.z, 3, 128, 128)), 'samples_{}'.format(frame), args.save_dir, individual=False, board=True)

def image_grad(image, hist_name):

    # get image gradient cdfs for generator, and real images
    print image.shape
    gmag = []
    for i in range(image.shape[0]):
        im = tf.reshape(image[i], [128, 128, 3])
        print im.shape
        im = im.eval()
        print im.shape
        gray = color.rgb2gray(im)
        # sobel derivative filters
        gx = np.zeros(gray.shape)
        filters.sobel(gray,1,gx)
        gy = np.zeros(gray.shape)
        filters.sobel(gray,0,gy)
        g_mag.append(np.sqrt(gx**2+gy**2))
    tf.summary.histogram(hist_name, g_mag)
    return g_mag

def inv_sample(data, n_samples):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cdf_values = np.zeros(bin_edges.shape)
    cdf_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cdf_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

#samples = generate_image_tf('', real_data_int, args.save_dir)
#summ_samples = tf.summary.image("generated samples", samples)

# For calculating inception score
samples_100 = G(100)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 128, 128)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Adversarial dataset generators
x = np.load(os.getcwd()+'/../toolkit/npy/adv_trial1.npy')
def adv_gen(x, batch_size):

    def get_epoch():
        np.random.shuffle(x)
        for i in xrange(len(x) / batch_size):
            feed = x[i*batch_size:(i+1)*batch_size]
            feed = np.reshape(feed, (-1, 49152))
            yield (np.copy(feed))

    return get_epoch

def adv_load(x, batch_size):

    return (adv_gen(x[:int(x.shape[0]*.8)],batch_size), adv_gen(x[int(x.shape[0]*.2):],batch_size))

train_gen_adv, dev_gen_adv = adv_load(x, BATCH_SIZE)

# Vanilla Cifar generators
train_gen1, dev_gen1 = lib.cub128.load_cub128(BATCH_SIZE*2, data_dir=DATA_DIR)
train_gen2, dev_gen2 = lib.cub128.load_cub128(BATCH_SIZE, data_dir=DATA_DIR)
#train_gen1, dev_gen1 = lib.cifar10.load(BATCH_SIZE*2, data_dir=DATA_DIR)
#train_gen2, dev_gen2 = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)

def inf_train_gen1():
    while True:
        for images in train_gen1():
            yield (images)

def inf_train_gen2():
    while True:
        for images in train_gen2():
            yield (images)

def inf_train_gen_adv():
    while True:
        for images, in train_gen_adv():
            yield (images)

# Train loop

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
d2 = InitD2()
args = load_args()
summaries_dir = './board'
summary_op = tf.summary.merge_all()

print args.save_dir

with tf.Session(config=config) as session:
    train_writer = tf.summary.FileWriter(summaries_dir + '/train/'+args.save_dir+'/'+args.tf_trial_name,
                                              session.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    session.run(tf.global_variables_initializer())
    gen1 = inf_train_gen1() # yields double in the beginning
    gen2 = inf_train_gen2()
    gen_adv = inf_train_gen_adv()
    gen_iters = GEN_ITERS
    d2_start = D2_START_ITERS
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('./models/vanilla_wgan-gp_20000_steps.meta')
    #saver.restore(session, './models/vanilla_wgan-gp_50000_steps')
    """
    for iteration in range(500):
        generate_random_image(iteration, args.save_dir)
    """
    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            for i in xrange(gen_iters):
                _ = session.run(gen_train_op,  feed_dict={ train_d2: 0 })

        # Train critic

        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            # second phase, we have trained G and D1, now train including D2
            if iteration < d2_start:
                _data, = gen1.next()
                _d1_cost, _ = session.run([d1_cost, d1_train_op], feed_dict={real_data_int: _data, train_d2: 0 })

           # first phase, just train on batches of 128 real images
            else:
                _data_adv, = gen_adv.next()

                _data, = gen2.next()

                # mix in the adversarial and real data
                _data = np.concatenate((_data, _data_adv))
                _d1_cost, _d2_cost, _, _ = session.run([d1_cost, d2_cost, d1_train_op, d2_train_op], feed_dict={real_data_int: _data, train_d2: 1 })

            """
            _d2_cost, _ = session.run([d2_cost, d2_train_op], feed_dict={real_data_int: _data_adv,
                                                                                                           y : _labels_adv})
            """
        lib.plot.plot('train d1-wgan cost', _d1_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration > d2_start:
            lib.plot.plot('train d2-bce cost', _d2_cost)
            lib.plot.plot('time', time.time() - start_time)


        # Calculate inception score every 1K iters
        if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_d1_costs = []
            dev_d2_costs = []

            if iteration < d2_start:
                 for images in dev_gen1():
                    images = np.reshape(images, (BATCH_SIZE*2, 3, 128, 128))
                    _dev_d1_cost, summary = session.run([d1_cost, summary_op], feed_dict={real_data_int:images,
                                                                                          train_d2: 0 })
                    dev_d1_costs.append(_dev_d1_cost)
                    lib.plot.plot('dev d1 cost', np.mean(dev_d1_costs))
                    """
                    mag_real = image_grad(real_data, "gradient magnitude real X")
                    mag_gen  = image_grad(gen_data, "gradient magnitude gen X")

                    cdf_real_samples = sample_cdf(mag_real)
                    cdf_gen_samples = sample_cdf(mag_gen)
                    """
                    if iteration > 50000:
                        generate_image(iteration, _data, args.save_dir)
                        generate_random_image(iteration, _data, args.save_dir+'/random')
            else:

                for images, in dev_gen2():
                    for images_adv, in dev_gen_adv():
                        _dev_d1_cost, _dev_d2_cost, summary = session.run([d1_cost, d2_cost, summary_op], feed_dict={real_data_int: images, train_d2: 1 })
                        dev_d1_costs.append(_dev_d1_cost)
                        dev_d2_costs.append(_dev_d2_cost)
                        break
                lib.plot.plot('dev d1 cost', np.mean(dev_d1_costs))
                lib.plot.plot('dev d2 cost', np.mean(dev_d2_costs))
                if iteration > 50000:
                    generate_image(iteration, _data_adv, args.save_dir)
                    generate_image(iteration, _data, args.save_dir)
                    generate_random_image(iteration, _data, args.save_dir+'/random')

            # for tensorboard
            """
            train_writer.add_summary(summary, iteration)
            samples = generate_image_tf('', real_data_int, args.save_dir)
            samples = tf.reshape(samples, [-1, samples.shape[0], samples.shape[1], samples.shape[2]])
            samples = tf.cast(samples, tf.float32)
            samples_op = tf.summary.image("generated samples", samples)
            summ_im = session.run(samples_op)
            train_writer.add_summary(summ_im, iteration)
            """
        if iteration in SAVE_ITERS:
            saver.save(session, 'models/'+SAVE_NAME+'_'+str(iteration)+'_steps')
        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
