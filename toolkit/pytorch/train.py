'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from models import load

from utils import progress_bar
from torch.autograd import Variable

def load_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume_from', '-r', default=None, help='resume from checkpoint')
    parser.add_argument('--model', '-m', default='vgg16', help='load model: <vgg16,vgg19,resnet18/34/50/101/152,googlenet,densenet')
    parser.add_argument('--n_epochs', '-e', default=200, type=int, help='number of epochs to train for')
    parser.add_argument('--pool_size', '-p', default=0, type=int, help='size of initial spatial pooling')

    args = parser.parse_args()

    return args

class Trainer(object):

    def __init__(self, args):

        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0
        # Load Data Transformations
        print('==> Preparing data..')
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.load_model()

    def load_model(self):
        # Model
        """if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume_from)

            self.net = checkpoint['net']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        """
        if False: pass

        else:
            print('==> Building model..')
            fn = getattr(load, self.args.model)
            if self.args.resume_from is not None:
                self.net = fn(self.args.pool_size, self.args.resume_from)
                load_state = OrderedDict()
                for k, v in state_dict['model'].items():
                    if self.use_cuda:
                        name = k[7:]
                    else:
                        name = k
                    load_state[name] = v
                self.net.load_state_dict(load_state)

            else:
                self.net = fn()

        print self.net

        if  self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=[0])
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=1e-4)

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(self, epoch):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            net_state_dict = self.net.module.state_dict() if self.use_cuda > 1 else self.net.state_dict()
	    state = {
                'model': net_state_dict,
                'epoch': epoch,
                'optim': self.optimizer,
                'acc'  : acc
            }
            """
            state = {
                'net':   self.net.module if self.use_cuda else self.net,
                'acc':   acc,
                'epoch': epoch,
            }
            """
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            self.best_acc = acc


if __name__ == '__main__':

    args = load_args()
    clf = Trainer(args)
    start = clf.start_epoch

    for epoch in range(start, int(args.n_epochs)):
        clf.train(epoch)
        clf.test(epoch)

