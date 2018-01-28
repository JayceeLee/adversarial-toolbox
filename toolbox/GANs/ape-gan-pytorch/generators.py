from torch import nn


class CIFARgenerator(nn.Module):
    def __init__(self, args):
        super(CIFARgenerator, self).__init__()
        self.shape = (32, 32, 3)
        self.dim = args.dim
        preprocess = nn.Sequential(
                nn.Linear(128, 4 * 4 * 4 * self.dim),
                nn.BatchNorm2d(4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
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
        self.dim = args.dim
        self.shape = (28, 28, 1)
        preprocess = nn.Sequential(
                nn.Linear(128, 4*4*4*self.dim),
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
        output = output.view(-1, 4*self.dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, 784)
