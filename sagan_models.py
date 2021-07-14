import torch
import numpy as np
import torch.nn as nn
from spectral import SpectralNorm


class Self_Attn(nn.Module):
    ''' Self attention Layer '''
    def __init__(self, in_channels, activation):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
            inputs:
                x: input feature maps (B x C x W x H)
            returns:
                out: self attention value + input feature 
                attention: B x N x N (N is width * height)
        '''
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1) # B x N x C
        proj_key = self.key_conv(x).view(B, -1, W * H) # B X C x N
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B x N x N
        proj_value = self.value_conv(x).view(B, -1, W * H) # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x N
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    ''' Generator '''
    def __init__(self, batch_size, im_size=64, z_dim=100, conv_dim=64, adv_loss='hinge'):
        super(Generator, self).__init__()
        self.im_size = im_size
        Normalization = SpectralNorm if adv_loss == 'hinge' else lambda x: x
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.im_size)) - 2
        mult = 2 ** repeat_num # 8
        layer1.append(Normalization(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult # 512
        layer2.append(Normalization(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2) # 256
        layer3.append(Normalization(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if self.im_size == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(Normalization(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)

        curr_dim = int(curr_dim / 2) # 128
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        if self.im_size == 64:
            out = self.l4(out)
            out, p2 = self.attn2(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    ''' Discriminator '''

    def __init__(self, batch_size=64, im_size=64, conv_dim=64, adv_loss='hinge'):
        super(Discriminator, self).__init__()
        self.im_size = im_size
        Normalization = SpectralNorm if adv_loss == 'hinge' else lambda x: x
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(Normalization(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim

        layer2.append(Normalization(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(Normalization(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if self.im_size == 64:
            layer4 = []
            layer4.append(Normalization(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        if self.im_size == 64:
            out = self.l4(out)
            out, p2 = self.attn2(out)
        out = self.last(out)
        return out.squeeze()
