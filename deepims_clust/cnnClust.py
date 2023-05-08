import torch.nn as nn
from typing import Literal

from .cae import conv2d_hwout

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


class CNNClust(nn.Module):

    def __init__(self, 
                 num_clust: int, 
                 height: int, 
                 width: int,
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 dropout: float=0.1
                ):
        super(CNNClust, self).__init__()
        self.num_clust = num_clust
        self.height = height
        self.width = width

        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout)
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout)
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout)
                                   )

        self.conv6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()
                                   )

        # Computing dimensions for linear layer
        l2h, l2w = conv2d_hwout(height=self.height, width=self.width, padding=(0, 0),
                                dilation=(1, 1), kernel_size=(2, 2), stride=(1, 1))
        l2hh, l2ww = conv2d_hwout(height=l2h, width=l2w, padding=(0, 0),
                                  dilation=(1, 1), kernel_size=(2, 2), stride=(2, 2))
        l3h, l3w = conv2d_hwout(height=l2hh, width=l2ww, padding=(0, 0),
                                dilation=(1, 1), kernel_size=(2, 2), stride=(1, 1))
        l4h, l4w = conv2d_hwout(height=l3h, width=l3w, padding=(0, 0),
                                dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))
        l5h, l5w = conv2d_hwout(height=l4h, width=l4w, padding=(0, 0),
                                dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))
        l5hh, l5ww = conv2d_hwout(height=l5h, width=l5w, padding=(0, 0),
                                  dilation=(1, 1), kernel_size=(2, 2), stride=(2, 2))
        l6h, l6w = conv2d_hwout(height=l5hh, width=l5ww, padding=(0, 0),
                                dilation=(1, 1), kernel_size=(3, 3), stride=(1, 1))

        self.final_conv_dim = l6h*l6w
        print(f'CNNClust final conv dim = {self.final_conv_dim}')
        
        if activation == 'softmax':
            self.fc = nn.Sequential(nn.Linear(self.final_conv_dim, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.Softmax(dim=1))
        elif activation == 'sigmoid':
            self.fc = nn.Sequential(nn.Linear(self.final_conv_dim, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.sigmoid())
        elif activation == 'relu':
            self.fc = nn.Sequential(nn.Linear(self.final_conv_dim, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.ReLU())
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')
        
        # Init lower weights
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, self.final_conv_dim)
        x = self.fc(x)

        return x
