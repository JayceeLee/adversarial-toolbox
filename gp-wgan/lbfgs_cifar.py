import os, sys
sys.path.append(os.getcwd())

import time
#import glob

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('pdf')
from skimage import color
from scipy.ndimage import filters

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.cifar10
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
DATA_DIR = os.getcwd()+'/../toolkit/cifar-10-batches-py'

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
GEN_ITERS = 1 # how many generator iterations per disc iteration (old DCGAN hack)
BATCH_SIZE = 32 # Batch size
BATCH_SIZE_ADV = 32 # Batch size
ITERS = 100000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
D2_START_ITERS = 50000
SAVE_ITERS = [1, 10000, 20000, 50000, 100000]
SAVE_NAME = "vanilla_wgan-gp"
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

def G(n_samples, noise=None):

    if noise is None:
        noise = tf.random_normal([n_samples, args.z])

    output = lib.ops.linear.Linear('G.Input', args.z, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('G.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('G.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('G.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('G.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('G.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('G.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

base_d = False

def D1(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('D1.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('D1.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('D1.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    # we relax the dimensionality constraint here. Make it big
    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('D1.Output', 4*4*4*DIM, 1, output)

    #print "d1 ", tf.reshape(output, [-1]).shape
    return tf.reshape(output, [-1])

def D2(inputs):
    """
    we need a second descriminator to fool. This one mirrors a binary
    adversarial detector. Given some sample I = G(x), we need to fool the first
    discriminator into thinking that I ~ P(x), and we also need to fool the binary
    detector into thinking that this new adversarial image comes from the natural manifold
    """
    inputs = tf.reshape(inputs, [-1, 32, 32, 3])
    d2 = detector(inputs)
    detector.summary()
    d2 = tf.reshape(d2, [-1])
    return d2

def get_lbfgs(loss):
  print_iterations = args.print_iterations if args.verbose else 0
  if args.optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      loss, method='L-BFGS-B',
      options={'maxiter': args.max_iterations,
                  'disp': print_iterations})
  return optimizer

def minimize_lbfgs(sess, net, optimizer, init_img):
  if args.verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  optimizer.minimize(sess)
# calculate image gradient magnitudes and plot histogram

# call with adversarials and with real image directories
def get_images(image_dir)
    imgs = []
    images = glob(image_dir+"/*.png")
    for im in images:
	img = imread(im)
	img = img.astype(np.float32)
	img = preprocess(img)
	imgs.append(img)
    return imgs

def stylize(content_img, label, init_img, frame=None):
  with tf.device(args.device), tf.Session() as sess:

    # setup network
    net = detector(img)
    base_net = classifier(img)
    # get new label
    new_label = 0
    while new_label is not label:
	new_label = np.random.randint(0, args.num_labels)

    print "original prediction: {}".format(label)
    print "Adversarial target:  {}".format(new_label)
    # image loss
    L_image = net(img)

    # content loss
    L_content = sum_content_losses(sess, net, content_img)

    # denoising loss
    L_tv = sum_total_variation_losses(sess, net, init_img)

    # loss weights
    alpha = args.content_weight
    beta  = args.style_weight
    theta = args.tv_weight

    # total loss
    L_total  = alpha * L_content
    L_total += beta  * L_style
    L_total += theta * L_tv

    # video temporal loss
    if args.video and frame > 1:
      gamma      = args.temporal_weight
      L_temporal = sum_shortterm_temporal_losses(sess, net, frame, init_img)
      L_total   += gamma * L_temporal

    # optimization algorithm
    optimizer = get_optimizer(L_total)

    if args.optimizer == 'adam':
      minimize_with_adam(sess, net, optimizer, init_img, L_total)
    elif args.optimizer == 'lbfgs':
      minimize_with_lbfgs(sess, net, optimizer, init_img)

    output_img = sess.run(net['input'])
    output_img = convert_to_original_colors(np.copy(content_img), output_img)

    write_image_output(output_img, content_img, style_imgs, init_img)

""" Here we grab samples from G and push them through D1 and D2 """
real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE*2, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
gen_data = G(BATCH_SIZE*2)
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE*2])
train_d2 = tf.placeholder(tf.int32)
test_summ = tf.placeholder(tf.int32)
"""
We need to make a decision regarding two distributions,
We constrain our manifold to only include images from the union on sets P1 and P2
P1 being the set of natural images that could come from Cifar
P2 being the set of adversarial images that can fool our detector
"""
disc_nat  = D2(real_data)
disc_adv  = D2(gen_data)

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
g_params = lib.params_with_name('G')
d1_params = lib.params_with_name('D1')
#d2_params = lib.params_with_name('D2')

# Standard WGAN loss on G and first discriminator
# Control flow on the cost function. If were not using D2 we only need to consider one discriminator in our loss
#gen_cost = tf.reduce_mean(disc_adv)

gen_cost = tf.cond(train_d2 > 0, lambda: -tf.reduce_mean(disc_fake)-tf.reduce_mean(disc_adv),
                                 lambda: -tf.reduce_mean(disc_fake))
d2_cost = tf.cond(train_d2 > 0,  lambda: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_adv, labels=y)),
                                 lambda: 0.)
d1_cost  =  tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

""" Gradient penalty """

# TODO Verify that the penalty on the gradient doesn't apply to D2

alpha = tf.random_uniform(
    shape=[BATCH_SIZE*2,1],
    minval=0.,
    maxval=1.
)
# calculate two sided loss term, mean( sqrt( sum ( gradx^2 ) ) )
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
                                      gen_cost, var_list=g_params
                                     )
d1_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                                       d1_cost, var_list=d1_params
                                     )

# we dont want to train the detector, just load weights
d2_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                                       d2_cost#, var_list=d2_params
                                     )
d1_summ  = tf.summary.scalar("d1 cost", d1_cost)
d2_summ  = tf.summary.scalar("d2 cost", d2_cost)
gen_summ = tf.summary.scalar("gen cost", gen_cost)

fixed_noise_z = tf.constant(np.random.normal(size=(args.z, args.z)).astype('float32'))
fixed_noise_samples_z = G(args.z, noise=fixed_noise_z)

def generate_random_image(frame,  save, board=False):
    random_noise_z = tf.constant(np.random.normal(size=(args.z, args.z)).astype('float32'))
    random_noise_samples_z = G(args.z, noise=random_noise_z)
    samples = session.run(random_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((args.z, 3, 32, 32)), 'samples_{}'.format(frame), save, individual=False)


def generate_image(frame, true_dist, save, board=False):
    samples = session.run(fixed_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((args.z, 3, 32, 32)), 'samples_{}'.format(frame), args.save_dir, individual=False)

def generate_image_tf(frame, true_dist, save):
    samples = session.run(fixed_noise_samples_z)
    samples = ((samples+1.)*(255./2)).astype('int32')
    return lib.save_images.save_images(samples.reshape((args.z, 3, 32, 32)), 'samples_{}'.format(frame), args.save_dir, individual=False, board=True)

def image_grad(image, hist_name):

    # get image gradient cdfs for generator, and real images
    print image.shape
    gmag = []
    for i in range(image.shape[0]):
        im = tf.reshape(image[i], [32, 32, 3])
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
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Adversarial dataset generators
x = np.load(os.getcwd()+'/../toolkit/npy/adv_trial1.npy')
def adv_gen(x, batch_size):

    def get_epoch():
        np.random.shuffle(x)
        for i in xrange(len(x) / batch_size):
            feed = x[i*batch_size:(i+1)*batch_size]
            feed = np.reshape(feed, (-1, 3072))
            yield (np.copy(feed), np.zeros(batch_size))

    return get_epoch

def adv_load(x, batch_size):

    return (adv_gen(x[:int(x.shape[0]*.8)],batch_size), adv_gen(x[int(x.shape[0]*.2):],batch_size))

train_gen_adv, dev_gen_adv = adv_load(x, BATCH_SIZE)

# Vanilla Cifar generators
train_gen1, dev_gen1 = lib.cifar10.load(BATCH_SIZE*2, data_dir=DATA_DIR)
train_gen2, dev_gen2 = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)

def inf_train_gen1():
    while True:
        for images, targets in train_gen1():
            yield (images, targets)

def inf_train_gen2():
    while True:
        for images, targets in train_gen2():
            yield (images, targets)

def inf_train_gen_adv():
    while True:
        for images, targets in train_gen_adv():
            yield (images, targets)

# Train loop

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
d2 = InitD2()
args = load_args()
summaries_dir = './board'
for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

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
    saver.restore(session, './models/vanilla_wgan-gp_50000_steps')

    for iteration in range(500):
        generate_random_image(iteration, args.save_dir)

    for iteration in xrange(20000, ITERS):
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
                _data, _labels = gen1.next()
                _d1_cost, _ = session.run([d1_cost, d1_train_op], feed_dict={real_data_int: _data,
                                                                                                  y: _labels,
                                                                                                  train_d2: 0 })

           # first phase, just train on batches of 128 real images
            else:
                _data_adv, _labels_adv = gen_adv.next()

                _data, _labels = gen2.next()
                _labels = np.ones(_data.shape[0])

                # mix in the adversarial and real data
                _data = np.concatenate((_data, _data_adv))
                _labels = np.concatenate((_labels, _labels_adv))
                _data, _labels = shuffle(_data, _labels)
                _d1_cost, _d2_cost, _, _ = session.run([d1_cost, d2_cost, d1_train_op, d2_train_op], feed_dict={real_data_int: _data,
                                                                                                                y : _labels,
                                                                                                                train_d2: 1 })

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
                 for images, labels in dev_gen1():
                    _dev_d1_cost, summary = session.run([d1_cost, summary_op], feed_dict={real_data_int:images,
                                                                     y: labels,
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

                for images, labels in dev_gen2():
                    for images_adv, labels_adv in dev_gen_adv():
                        labels = np.ones(images.shape[0])
                        images = np.concatenate((images, images_adv))
                        labels = np.concatenate((labels, labels_adv))
                        images, labels = shuffle(images, labels)
                        _dev_d1_cost, _dev_d2_cost, summary = session.run([d1_cost, d2_cost, summary_op], feed_dict={real_data_int: images,
                                                                                                                     y: labels,
                                                                                                                     train_d2: 1 })
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
