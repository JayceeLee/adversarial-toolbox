from tflib.ops.conv2d import Conv2D
from tflib.ops.linear import Linear
from tflib.ops.res import ResidualBlock
from tflib.ops.res import OptimizedResBlockDisc1
from tflib.ops.res import LeakyReLU, nonlinearity
from cifar_model import vggbn
import keras.backend as K
import tensorflow as tf

def InitD2():

    detector = vggbn(top='detector')
    detector.load_weights('../models/lbfgs/lbfgs_cifar.h5', by_name=True)
    for layer in detector.layers:
        layer.trainable=False
    K.set_learning_phase(0)
    return detector

def DCifar(inputs, dim):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = Conv2D('D1.1', 3, dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('D1.2', dim, 2*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('D1.3', 2*dim, 4*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    # we relax the dimensionality constraint here. Make it big
    output = tf.reshape(output, [-1, 4*4*4*dim])
    output = Linear('D1.Output', 4*4*4*dim, 1, output)
    return tf.reshape(output, [-1])

def DCifarResnet(inputs, labels, dim, conditional=True, acgan=True):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', dim, dim, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', dim, dim, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', dim, dim, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = Linear('Discriminator.Output', dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if conditional and acgan:
        output_acgan = Linear('Discriminator.ACGANOutput', dim, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None

def DResnet(inputs, dim, dim_out):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

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
    output = Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])

def DMnist(inputs, dim):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = Conv2D('Discriminator.1',1,dim,5,output,stride=2)
    output = LeakyReLU(output)
    output = Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.reshape(output, [-1, 4*4*4*dim])
    output = Linear('Discriminator.Output', 4*4*4*dim, 1, output)

    return tf.reshape(output, [-1])
