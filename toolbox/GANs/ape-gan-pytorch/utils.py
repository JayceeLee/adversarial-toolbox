import torch
import torch.autograd as autograd
import numpy as np
from data import mnist
from data import cifar10
from scipy.misc import imsave
from ppmimage import show as ppmshow


def dataset_iterator(args):
    if args.dataset == 'mnist':
        train_gen, dev_gen, test_gen = mnist.load(args.batch_size, args.batch_size)
    if args.dataset == 'cifar10':
        data_dir = '../../../images/cifar-10-batches-py/'
        train_gen, dev_gen = cifar10.load(args.batch_size, data_dir)
        test_gen = None

    return (train_gen, dev_gen, test_gen)


def inf_train_gen(train_gen):
    while True:
        for images, _ in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


def generate_image(iter, model, save_path, batch_size):
    datashape = model.shape
    if datashape == (28, 28, 1):
        fixed_noise_128 = torch.randn(batch_size, 128).cuda()
    else:
        fixed_noise_128 = torch.randn(128, 128).cuda()
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    if datashape == (28, 28, 1):
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *datashape)
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
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
    
    ppmshow(img)
    
    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)


