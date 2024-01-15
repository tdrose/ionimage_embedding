from re import S
from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Tuple, Literal, Final

from ..crl.dims import conv2d_hout, conv2d_wout

# Constants
D1: Final[int] = 8
D2: Final[int] = 16
K1: Final[Tuple[int, int]] = (3, 3)
K2: Final[Tuple[int, int]] = (3, 3)
S1: Final[Tuple[int, int]] = (2, 2)
S2: Final[Tuple[int, int]] = (3, 3)


class Encoder(nn.Module):
    def __init__(self, height: int, width: int, n_classes: int, encoder_dim: int=7,
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax'):

        super(Encoder, self).__init__()

        self.k1: Tuple[int, int] = K1
        self.k2: Tuple[int, int] = K2
        self.s1: Tuple[int, int] = S1
        self.s2: Tuple[int, int] = S2
        self.height = height
        self.width = width
        self.d1, self.d2 = D1, D2

        # Layers
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.d1, kernel_size=self.k1, 
                                             stride=self.s1, padding=(0, 0),
                                             dilation=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(self.d1, momentum=0.01),
                                   nn.ReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(self.d1, self.d2, kernel_size=self.k2, 
                                             stride=self.s2, padding=(0, 0),
                                             bias=False),
                                   nn.BatchNorm2d(self.d2, momentum=0.01),
                                   nn.ReLU())
        
        self.l1height = conv2d_hout(height=height, padding=(0, 0), dilation=(1, 1),
                                    kernel_size=self.k1, stride=self.s1)
        self.l1width = conv2d_wout(width=width, padding=(0, 0), dilation=(1, 1),
                                   kernel_size=self.k1, stride=self.s1)
        self.l2height = conv2d_hout(height=self.l1height, padding=(0, 0), dilation=(1, 1),
                                    kernel_size=self.k2, stride=self.s2)
        self.l2width = conv2d_wout(width=self.l1width, padding=(0, 0), dilation=(1, 1),
                                   kernel_size=self.k2, stride=self.s2)
        
        self.final_conv_dim = (self.l2height * self.l2width * self.d2) + n_classes

        
        self.h1 = self.final_conv_dim // 2
        self.h2 = self.final_conv_dim // 4
        
        
        if self.h1 < encoder_dim:
            self.h1 = encoder_dim
        if self.h2 < encoder_dim:
            self.h2 = encoder_dim
        
        self.lh1 = nn.Sequential(nn.Linear(self.final_conv_dim, self.h1),
                                 nn.BatchNorm1d(self.h1, momentum=0.01),
                                 nn.ReLU())
        
        if activation == 'softmax':
            self.final = nn.Sequential(nn.Linear(self.h1, encoder_dim),
                                    nn.BatchNorm1d(encoder_dim, momentum=0.01),
                                    nn.Softmax(dim=1))
        elif activation == 'sigmoid':
            self.final = nn.Sequential(nn.Linear(self.h1, encoder_dim),
                                    nn.BatchNorm1d(encoder_dim, momentum=0.01),
                                    nn.Sigmoid())
        elif activation == 'relu':
            self.final = nn.Sequential(nn.Linear(self.h1, encoder_dim),
                                    nn.BatchNorm1d(encoder_dim, momentum=0.01),
                                    nn.ReLU())
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')
        

        print(f'CVAEencoder final conv dim = {self.final_conv_dim}')
        print(f'CVAEencoder h1 dim = {self.h1}')
        print(f'CVAEencoder h2 dim = {self.h2}')


    def forward(self, x, c):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # vectorize x
        x = x.view(x.size(0), -1)

        # print(x.shape)
        # print(c.shape)
        # concatenate x and c
        x = torch.cat((x, c), dim=1)

        x = self.lh1(x)
        x = self.final(x)

        return x

class Decoder(nn.Module):
    def __init__(self, height: int, width: int, n_classes: int, encoder_dim: int=7,
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 encoder_l1height: int=0, encoder_l1width: int=0, encoder_l2height: int=0,
                 encoder_l2width: int=0):
        
        super(Decoder, self).__init__()

        self.k1: Tuple[int, int] = K1
        self.k2: Tuple[int, int] = K2
        self.s1: Tuple[int, int] = S1
        self.s2: Tuple[int, int] = S2

        self.encoder_l1height = encoder_l1height
        self.encoder_l1width = encoder_l1width
        self.encoder_l2height = encoder_l2height
        self.encoder_l2width = encoder_l2width

        self.height = height
        self.width = width
        self.d1, self.d2 = D1, D2

        # Layers
        self.h1 = int(encoder_dim * 2)
        self.h2 = int(encoder_dim * 4)

        if activation == 'softmax':
            self.fc1 = nn.Sequential(nn.Linear(encoder_dim+n_classes, self.h1),
                                    nn.BatchNorm1d(self.h1, momentum=0.01),
                                    nn.Softmax(dim=1))
        elif activation == 'sigmoid':
            self.fc1 = nn.Sequential(nn.Linear(encoder_dim+n_classes, self.h1),
                                    nn.BatchNorm1d(self.h1, momentum=0.01),
                                    nn.Sigmoid())
        elif activation == 'relu':
            self.fc1 = nn.Sequential(nn.Linear(encoder_dim+n_classes, self.h1),
                                    nn.BatchNorm1d(self.h1, momentum=0.01),
                                    nn.ReLU())
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')
        
        self.fc2 = nn.Sequential(nn.Linear(self.h1, 
                                           self.d2*self.encoder_l2height*self.encoder_l2width),
                                 nn.BatchNorm1d(self.d2*self.encoder_l2height*self.encoder_l2width, 
                                                momentum=0.01),
                                 nn.ReLU())

        self.ct1 = nn.ConvTranspose2d(in_channels=self.d2, out_channels=self.d1, 
                                      kernel_size=self.k2,
                                      stride=self.s2, padding=(0, 0), dilation=(1, 1),
                                      output_padding=(1, 1))
        self.tbn1 = nn.BatchNorm2d(self.d1, momentum=0.01)
        self.trelu1 = nn.ReLU()
        self.ct2 = nn.ConvTranspose2d(in_channels=self.d1, out_channels=1, kernel_size=self.k1,
                                      stride=self.s1, padding=(0, 0), dilation=(1, 1),
                                      output_padding=(1, 1))
        self.tbn2 = nn.BatchNorm2d(1, momentum=0.01)
        self.trelu2 = nn.ReLU()

    
    def convtrans1(self, x, output_size):
        x = self.ct1(x, output_size=output_size)
        x = self.tbn1(x)
        x = self.trelu1(x)
        return x

    def convtrans2(self, x, output_size):
        x = self.ct2(x, output_size=output_size)
        x = self.tbn2(x)
        x = self.trelu2(x)
        return x
    
    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.d2, self.encoder_l2height, self.encoder_l2width)
        x = self.convtrans1(x, output_size=(self.encoder_l1height, self.encoder_l1height))
        x = self.convtrans2(x, output_size=(self.height, self.width))
        x = functional.interpolate(x, size=(self.height, self.width), mode='bilinear')

        return x
