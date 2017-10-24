import tensorflow as tf
from layernorm import Layernorm
from cond_batchnorm import Batchnorm as CBatchnorm
from batchnorm import Batchnorm
from linear import Linear
from conv2d import Conv2D
from deconv2d import Deconv2D
import functools


def nonlinearity(x):
    return tf.nn.relu(x)


def prelu(x, name=None):
    with tf.variable_scope(name) as scope:
        try:
            alphas = tf.get_variable(name, x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
        except ValueError:
            scope.reuse_variables()
            alphas = tf.get_variable(name)

    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.3
    return pos + neg


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(name, inputs, labels=None, conditional=False, acgan=False):
    if not conditional:
        labels = None
    if conditional and acgan and ('Discriminator' in name):
        labels = None

    if 'Generator' in name:
        if labels is not None:
            return CBatchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=10)
        else:
            return Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def NormalizeD(name, axes, inputs):
    if ('Discriminator' in name):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return Layernorm(name, [1, 2, 3], inputs)
    else:
        return Batchnorm(name, axes, inputs, fused=True)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)


def MaxPool(input, size, stride, padding):
    output = tf.nn.max_pool(input, size, stride, padding, data_format='NCHW')
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, stride=1):
    output = Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases, stride=stride)
    output = tf.add_n([output[:, :, ::2, ::2],
                       output[:, :, 1::2, ::2],
                       output[:, :, ::2, 1::2],
                       output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, stride=1):
    output = inputs
    output = tf.add_n([output[:, :, ::2, ::2],
                       output[:, :, 1::2, ::2],
                       output[:, :, ::2, 1::2],
                       output[:, :, 1::2, 1::2]]) / 4.
    output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, stride=stride)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, stride=1):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, stride=stride)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, conditional=True, acgan=True, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = Conv2D
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels, conditional, acgan)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels, conditional, acgan)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def UResBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None, skip=None, no_norm=False, stride=1):
    """
    resample: None, 'down', or 'up'
    Acts as Identity, Deconv, or Conv operations
    """
    if resample == 'down':
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim, stride=stride)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim, stride=stride)
    elif resample == None:
        conv_shortcut = Conv2D
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=output_dim, stride=stride)
        conv_2        = functools.partial(Conv2D, input_dim=output_dim, output_dim=output_dim, stride=stride)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample==None:  # Identity skip connection
        shortcut = inputs
    else:  # skip with a convolution in path
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs, stride=stride)

    output = inputs
    output = Normalize(name+'.N1', output)

    if skip is not None:
        skip_layer = skip
        # print 'skip layer shape: {}'.format(skip_layer.shape)
        # print 'concat with output shape: {}'.format(output.shape)
        output = output + skip_layer
        # output = tf.concat([output, skip_layer], axis=1)
        output = prelu(output, name+'.skip.prelu')
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
        if no_norm:
            output = Normalize(name+'.N2', output)
        # print 'making a final output of :', output.shape
        return output

    else:
        output = nonlinearity(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
        output = Normalize(name+'.N2', output)
        output = prelu(output, name+'.output.prelu')
        output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)
        return shortcut + output


def OptimizedResBlockDisc1(name, inputs, dim, input_dim=3, stride=1):
    conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=dim, stride=stride)
    conv_2        = functools.partial(ConvMeanPool, input_dim=dim, output_dim=dim, stride=1)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=dim, filter_size=1, he_init=False, biases=True, inputs=inputs, stride=stride)

    output = inputs
    output = conv_1(name+'.Conv1', filter_size=3, inputs=output)
    output = prelu(output, name+'.prelu')
    output = conv_2(name+'.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, conditional=True, acgan=True, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(Conv2D, stride=2)
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = Conv2D
        conv_1        = functools.partial(Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample==None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = NormalizeD(name+'.BN', [0, 2, 3], output)

    return shortcut + (0.3*output)
