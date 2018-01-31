import numpy as np
from torch import nn
from torch.nn import functional as F

class CIFARgenerator(nn.Module):
    def __init__(self, args):
        super(CIFARgenerator, self).__init__()
        self._name = 'cifarG'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 4 * 4 * 4 * self.dim),
                #nn.BatchNorm2d(4 * 4 * 4 * self.dim),
                nn.ReLU(True)
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
                nn.Dropout(p=0.3),
                #nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True)
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.Dropout(p=0.3),
                #nn.BatchNorm2d(self.dim),
                nn.ReLU(True)
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class MNISTgenerator(nn.Module):
    def __init__(self, args):
        super(MNISTgenerator, self).__init__()
        self._name = 'mnistG'
        self.dim = args.dim
        self.in_shape = int(np.sqrt(args.dim))
        self.shape = (self.in_shape, self.in_shape, 1)
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 4*4*4*self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(2*self.dim, self.dim, 5),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = output.view(-1, 4*self.dim, 4, 4)
        output = self.block1(output)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, 784)

class SRResNet(nn.Module):
    def __init__(self, args, shape):
        super(SRResNet, self).__init__()
        BN = args.batchnorm
        self._name = 'SRResNet'
        self.dim = args.dim
        self.n_resblocks = 16
        self.shape = shape
        self.factor = args.downsample

        convblock_init = nn.Sequential(
                nn.Conv2d(3, 64, 9, stride=1, padding=4),
                nn.PReLU()
                )

        convbn = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64)
                )
               
        upsample = nn.Sequential(
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 3, 9, stride=1, padding=4)
                )

        resblock = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64)
                )
        
        self.convbn = convbn
        self.convblock = convblock_init
        self.upsample = upsample
        self.resblock = resblock

    def forward(self, input):
        init = self.convblock(input)
        res_out = self.resblock(init) 
        res_out = res_out + init
        for i in range(self.n_resblocks):
            res_out = self.resblock(res_out) + res_out
        out = self.convbn(res_out) + init
        output = self.upsample(out)
        out_dim = self.factor * self.shape[1]
        output = output.view(-1, 3, out_dim, out_dim)
        return output
