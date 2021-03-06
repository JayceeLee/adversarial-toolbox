import torch
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
import numpy as np
from data import Mnist
from data import Cifar10
from data import Imagenet
from data import Raise
from scipy.misc import imsave
import matplotlib.pyplot as plt


def dataset_iterator(args):
    if args.dataset == 'mnist':
        train_gen, dev_gen, test_gen = Mnist.load(args.batch_size, args.batch_size)
    if args.dataset == 'cifar10':
        data_dir = '../../../images/cifar-10-batches-py/'
        train_gen, dev_gen = Cifar10.load(args.batch_size, data_dir)
        test_gen = None
    if args.dataset == 'imagenet':
        data_dir = '../../../images/imagenet12/imagenet_val_png/'
        train_gen, dev_gen = Imagenet.load(args.batch_size, data_dir)
        test_gen = None
    if args.dataset == 'raise':
        data_dir = '../../../images/raise/'
        train_gen, dev_gen = Raise.load(args.batch_size, data_dir)
        test_gen = None
    else:
        raise ValueError

    return (train_gen, dev_gen, test_gen)


def inf_train_gen(train_gen):
    while True:
        for images, _ in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


def stack_data(args, _data):
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
        )

    if args.task == 'sr':
        if args.dataset == 'cifar10':
            datashape = (3, 32, 32)
            _data = _data.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
            real_data = torch.stack([preprocess(item) for item in _data]).cuda()

        if args.dataset == 'imagenet':
            datashape = (3, 224, 224)
            _data = _data.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
            real_data = torch.stack([preprocess(item) for item in _data]).cuda(0)

    else:    
        if args.dataset == 'mnist':
            real_data = torch.Tensor(_data).cuda()

    return real_data

def scale_data(args, _data, downsample=True):
    tfs = []
    tfs.append(transforms.ToPILImage())
    tfs.append(transforms.RandomCrop(96))
    if downsample:
        new_dim = (24, 24)
        tfs.append(transforms.Resize(new_dim))
    tfs.append(transforms.ToTensor())
    scale = transforms.Compose(tfs)
    data = torch.stack([scale(item) for item in _data])
    if downsample:
        for i, x in enumerate(data):
            data[i] = ((data[i] - torch.min(data[i]))/
                    (torch.max(data[i]) - torch.min(data[i]))
                    )
    else:
        for i, x in enumerate(data):
            data[i] = (data[i]*2) - 1.

    return data.cuda(0)


def scale_data2(args, data):

    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomCrop(24*4),
        transforms.ToTensor()])

    normalize = transforms.Normalize(mean = [.5, .5, .5],
            std = [.5, .5, .5])

    scale = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(24),
        transforms.ToTensor(),
        transforms.Normalize(mean = [.5, .5, .5],
            std = [.5, .5, .5])
        ]) 
    low_res = torch.FloatTensor(args.batch_size, 3, 24, 24)
    high_res = torch.FloatTensor(args.batch_size, 3, 96, 96)

    for i in range(len(data)):
        x = transform(data[i])
        low_res[i] = scale(x)
        high_res[i] = normalize(x)
    lr = low_res.cuda(0)
    hr = high_res.cuda(0)
    
    return lr, hr

 
def generate_sr_image(iter, netG, save_path, args, data):
    lr, hr, sr = data
    batch_size = args.batch_size
    if netG._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    lr = lr.cpu().numpy().transpose(0, 2, 3, 1)[..., ::-1]
    hr = hr.cpu().numpy().transpose(0, 2, 3, 1)[..., ::-1]
    sr = sr.cpu().data.numpy().transpose(0, 2, 3, 1)[..., ::-1]
    show_sr(lr[0], hr[0], sr[0])


def generate_image(iter, model, save_path, args):
    batch_size = args.batch_size
    datashape = model.shape
    if model._name == 'mnistG':
        fixed_noise_128 = torch.randn(batch_size, args.dim).cuda()
    else:
        fixed_noise_128 = torch.randn(128, args.dim).cuda()
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    if model._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))


def show_sr(lr, hr, sr):
    lr = (lr - np.min(lr))/(np.max(lr) - np.min(lr))
    hr = (hr - np.min(hr))/(np.max(hr) - np.min(hr))
    sr = (sr - np.min(sr))/(np.max(sr) - np.min(sr))
    #print ("LR: ", np.max(lr), np.min(lr), lr.shape)
    #print ("HR: ", np.max(hr), np.min(hr), hr.shape)
    #print ("SR: ", np.max(sr), np.min(sr), sr.shape)
    plt.ion()
    #plt.figure()
    plt.suptitle("LR, HR, SR")
    plt.subplot(1, 3, 1)
    plt.imshow((lr+1))
    plt.subplot(1, 3, 2)
    plt.imshow((hr+1))
    plt.subplot(1, 3, 3)
    plt.imshow(sr/2)
    plt.draw()
    plt.pause(0.001)


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    #plt.imshow(img, cmap='gray')
    #plt.draw()
    #plt.pause(0.001)

    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)


