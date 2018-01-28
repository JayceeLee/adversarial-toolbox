import os
import sys
import time
import argparse
import numpy as np
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from scipy.misc import imshow

import ops
import plot
import utils
import generators
import discriminators
from data import mnist
from data import cifar10

def load_args():

    parser = argparse.ArgumentParser(description='recover-gan')
    parser.add_argument('-d', '--dim', default=64, type=int, help='latent space')
    parser.add_argument('-l', '--gp', default=10, type=int, help='grad penalty')
    parser.add_argument('-b', '--batch_size', default=50, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=784, type=int)
    parser.add_argument('--dataset', default='mnist')
    args = parser.parse_args()
    return args

def load_models(args):
    if args.dataset == 'mnist':
        netG = generators.MNISTgenerator(args).cuda()
        netD = discriminators.MNISTdiscriminator(args).cuda()

    if args.dataset == 'cifar10':
        netG = generators.CIFARgenerator(args).cuda()
        netD = discriminators.CIFARdiscriminator(args).cuda()
	
    print (netG, netD)
    return (netG, netD)


def train():
    args = load_args()
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netG, netD = load_models(args)

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()

    gen = utils.inf_train_gen(train_gen)

    preprocess = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for iteration in range(args.epochs):
        start_time = time.time()

        """ Update D network """
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for i in range(5):
            _data = next(gen)
            # imshow(_data[0].reshape((28, 28)))
            """train with real data"""
            if args.dataset != 'mnist':
                datashape = netG.shape[::-1]
                netD.zero_grad()        
                _data = _data.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
                real_data = torch.stack([preprocess(item) for item in _data]).cuda()
                real_data_v = autograd.Variable(real_data)
            else:
                real_data = torch.Tensor(_data).cuda()
                real_data_v = autograd.Variable(real_data)
                netD.zero_grad()

            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            """ train with fake """
            noise = torch.randn(args.batch_size, 128).cuda()
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            """ train with gradient penalty """
            gradient_penalty = ops.calc_gradient_penalty(args,
                    netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        """ Update G network """
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        noise = torch.randn(args.batch_size, 128).cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        """ Write logs and save samples """
        save_dir = './plots/'+args.dataset
        plot.plot(save_dir+'train disc cost', D_cost.cpu().data.numpy())
        plot.plot(save_dir+'/time', time.time() - start_time)
        plot.plot(save_dir+'/train gen cost', G_cost.cpu().data.numpy())
        plot.plot(save_dir+'/wasserstein distance', Wasserstein_D.cpu().data.numpy())

        """ Calculate dev loss and generate samples every 100 iters """
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen():
                if args.dataset != 'mnist':
                    images = images.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
                    imgs = torch.stack([preprocess(item) for item in images]).cuda()
                else:
                    imgs = torch.Tensor(images).cuda()
                imgs_v = autograd.Variable(imgs, volatile=True)
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            plot.plot(save_dir+'/dev disc cost', np.mean(dev_disc_costs))

            utils.generate_image(iteration, netG, save_dir, args.batch_size)

        """ Save logs every 100 iters """
        if (iteration < 5) or (iteration % 100 == 99):
            plot.flush()
        plot.tick()

if __name__ == '__main__':
    train()
