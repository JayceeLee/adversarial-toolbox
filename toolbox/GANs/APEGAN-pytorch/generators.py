import numpy as np
from torch import nn
from torch.nn import functional as F


class convblock(nn.module):
    def __init__(self, in_c, out_c, strides=(1, 1)):
        super(convblock, self).__init__()
        self.stride1 = strides[0]
        self.stride2 = strides[1]

        block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=self.stride1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, stride=self.stride2, padding=1),
                # nn.BatchNorm2d(out_ch), 
                nn.LeakyReLU(inplace=True)
                )
        self.block = block
    
    def forward(self, x):
        x = self.block(x)
        return x


class upblock(nn.Module):
    def __init__(self, in_c, out_c, bilinear=True):
        super(upblock, self).__init__()

        if bilinear:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upsample = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)

        self.convblock = convblock(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        dx = x1.size()[2] - x2.size()[2]
        dy = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (dx//2, int(dx/2), dy//2, int(dy/2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.convblock(x)
        return x


class UNet(nn.Module):
    def __init__(self, args, shape):
        super(UNet, self).__init__()
        self._name = 'unetG'
        self.shape = shape
        
        self.init = convblock(3, 64)
        self.enc1 = convblock(63, 128, (2, 1))
        self.enc2 = convblock(128, 256, (2, 1))
        self.enc3 = convblock(256, 512, (2, 1))
        self.enc4 = convblock(512, 512, (2, 1))
        self.dec4 = upblock(1024, 256)
        self.dec3 = upblock(512, 128)
        self.dec2 = upblock(256, 64)
        self.dec1 = upblock(128, 64)
        self.out = convblock(64, 3)

    def forward(self, x):
        x1 = self.init(x)
        x2 = self.enc1(x)
        x3 = self.enc2(x)
        x4 = self.enc3(x)
        x5 = self.enc4(x)
        x = self.dec4(x5, x4)
        x = self.dec4(x, x3)
        x = self.dec4(x, x2)
        x = self.dec4(x, x1)
        out = self.out(x)
        return out.view(-1, 3, 224, 224)
