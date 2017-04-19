import vgg
import resnet
import googlenet
import densenet


def vgg16(pretrained=None, **kwargs):
    return vgg.vgg16_bn(pretrained, **kwargs)

def vgg19(pretrained=None, **kwargs):
    return vgg.vgg19_bn(pretrained, **kwargs)

def vgg16_p(p, pretrained=None, **kwargs):
    return vgg.vgg16_bn_p(p, pretrained, **kwargs)

def vgg19_p(p, pretrained=None, **kwargs):
    return vgg.vgg19_bn_p(p, pretrained, **kwargs)

def resnet18():
    return resnet.resnet18()

def resnet35():
    return resnet.resnet34()

def resnet50():
    return resnet.resnet50()

def resnet101():
    return resnet.resnet101()

def resnet150():
    return resnet.resnet152()

def googlenet():
    return googlenet.GoogLeNet()

def densenet121():
    return densenet.densenet121()

def densenet169():
    return densenet.densenet169()

def densenet201():
    return densenet.densenet201()

def densenet():
    return densenet.densenet_cifar()



