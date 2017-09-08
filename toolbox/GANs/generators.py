
import tensorflow as tf
from tflib.ops.res import ResidualBlock, BottleneckResidualBlock
from tflib.ops.res import Normalize, nonlinearity
from tflib.ops.linear import Linear
from tflib.ops.batchnorm import Batchnorm
from tflib.ops.conv2d import Conv2D
from tflib.ops.deconv2d import Deconv2D


def CifarResnet(n_samples, dim, dim_out, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = Linear('Generator.Input', 128, 4*4*dim_out, noise)
    output = tf.reshape(output, [-1, dim, 4, 4])
    output = ResidualBlock('Generator.1', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', dim, dim, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', dim, dim, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = Conv2D('Generator.Output', dim, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])

def GCifar(n_samples, dim, dim_out, n, noise=None):

    if noise is None:
        noise = tf.random_normal([n_samples, n])
    output = Linear('G.Input', n, 4*4*4*dim, noise)
    output = Batchnorm('G.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*dim, 4, 4])
    output = Deconv2D('G.2', 4*dim, 2*dim, 5, output)
    output = Batchnorm('G.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Deconv2D('G.3', 2*dim, dim, 5, output)
    output = Batchnorm('G.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Deconv2D('G.5', dim, 3, 5, output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])

def GResnetOptim(n_samples, dim, dim_out, noise=None, nonlinearity=tf.nn.relu):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, dim_out])

def GResnet(n_samples, dim, dim_out, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, dim_out])

