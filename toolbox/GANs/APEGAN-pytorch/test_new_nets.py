import pretrainedmodels
import pretrainedmodels.utils as utils
import numpy as np
import torch.nn
import torch.autograd as autograd
import torchvision.transforms as transforms
from scipy.misc import imresize, imread, imshow
from glob import glob
from nets import inception_resnet_v2
from nets import torchvision_models as models

data_dir = '/home/neale/repos/adversarial-toolbox/images/adversarials/mim/imagenet/symmetric/inception_resnet_v2/'

gpu = 1

def rerange(x):
    return (x-np.min(x))/(np.max(x)-np.min(x)) 


def transform(x):
    # x = np.array([rerange(item) for item in x])
    x = x.transpose(0, 3, 1, 2)
    x = torch.stack([torch.Tensor(item) for item in x])
    return x
           

def load_path_npy(paths, arr, start=0, end=0):

    assert arr.ndim == 4
    imshape = (arr.shape[1], arr.shape[2], arr.shape[3])
    for idx, i in enumerate(range(start, end)):
        image = np.load(paths[idx])
        arr[i] = image
    print ("Loaded {} images".format(len(paths)))
    return arr


def load_npy(real, adv, n, shape=299):
    if n is None:
        n = len(real) - 1
    paths_real = glob(real+'_npy/*.npy')
    print (real+'_npy')
    print (paths_real[0])
    print (paths_real[1][106:120])
    paths_real.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f[106:120])))))
    paths_adv = glob(adv+'_npy/*.npy')
    paths_adv.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f[106:120])))))
    paths_real = paths_real[:n]
    paths_adv = paths_adv[:n]
    x_real = np.empty((len(paths_real), shape, shape, 3))
    real = load_path_npy(paths_real, arr=x_real, start=0, end=len(paths_real))
    x_adv = np.empty((len(paths_adv), shape, shape, 3))
    adv = load_path_npy(paths_adv, arr=x_adv, start=0, end=len(paths_adv))
    return real, adv


def test():
    real_dir = data_dir + 'real'
    adv_dir = data_dir + 'adv'
    real, adv = load_npy(real_dir, adv_dir, 200)
    print (real.shape, adv.shape)

    model = pretrainedmodels.__dict__['inceptionresnetv2'](
        num_classes=1001, pretrained='imagenet+background').cuda(gpu)
    model.eval()
    """ 
    load_img = utils.LoadImage()
    tf_img = utils.TransformImage(model)
    images = glob('/home/neale/repos/adversarial-toolbox/images/imagenet12/imagenet299/*.png')
    for path in images:
        img = load_img(path)
        in_tensor = tf_img(img)
        in_tensor = in_tensor.unsqueeze(0).cuda(gpu)
        input = autograd.Variable(in_tensor, requires_grad=False)
        print (np.argmax(model(input).data.cpu().numpy()))
    import sys
    sys.exit(0)
    """
    adv = transform(adv)
    real = transform(real)

    for p in model.parameters():
        p.requires_grad = False
    
    hits = 0.
    total_norm = 0.
    for (r, a) in zip(real, adv):
        test_r = r.unsqueeze(0)
        test_a = a.unsqueeze(0)
        total_norm += np.linalg.norm(a - r)
        test_r_v = autograd.Variable(test_r).cuda(gpu)
        test_a_v = autograd.Variable(test_a).cuda(gpu)
        y = test_r_v.data.cpu().numpy()
        x = test_a_v.data.cpu().numpy()
        lr = np.argmax(model(test_r_v).data.cpu().numpy())
        la = np.argmax(model(test_a_v).data.cpu().numpy())
        
        if lr != la:
            hits += 1

    rate = (hits / len(real)) * 100.

    print ("Adversarial Examples: {}% of {} images".format(rate, len(real)))

test()
   
