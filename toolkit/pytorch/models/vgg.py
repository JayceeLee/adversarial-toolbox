'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16P':['ML', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19P':['ML', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, initial_pool_size=2):
        super(VGG, self).__init__()

        self.features = self._make_layers(cfg[vgg_name], initial_pool_size)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, pool_size):
        layers = []
        in_channels = 3
        for x in cfg:

            if x == 'ML':
                layers += [nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)]

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def msr_init(net):
    '''Initialize layer parameters.'''
    for layer in net:
	if type(layer) == nn.Conv2d:
	    n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
	    layer.weight.data.normal_(0, math.sqrt(2./n))
	    layer.bias.data.zero_()
	elif type(layer) == nn.BatchNorm2d:
	    layer.weight.data.fill_(1)
	    layer.bias.data.zero_()
	elif type(layer) == nn.Linear:
	    layer.bias.data.zero_()

def vgg16_bn(pretrained, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG('VGG16')
    msr_init(model.features)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    print model
    return model

def vgg19_bn(pretrained, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG('VGG19')
    msr_init(model.features)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    print model
    return model

def vgg16_bn_p(pool_size, pretrained, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG('VGG16P', pool_size)
    msr_init(model.features)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    print model
    return model

def vgg19_bn_p(pool_size, pretrained, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG('VGG19P', pool_size)
    msr_init(model.features)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    print model
    return model
