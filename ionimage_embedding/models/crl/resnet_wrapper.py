import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal

class ResNetWrapper(torch.nn.Module):

    fc: nn.Sequential

    def __init__(self, 
                 num_clust: int,
                 height: int, width: int,
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 resnet: Literal['resnet18', 'resnet34', 'resnet50', 
                                 'resnet101', 'resnet152'] = 'resnet18',
                 pretrained: bool = False
                 ):

        super(ResNetWrapper, self).__init__()

        if resnet == 'resnet18':
            rn: models.ResNet = models.resnet18(pretrained=pretrained)
        elif resnet == 'resnet34':
            rn: models.ResNet = models.resnet34(pretrained=pretrained)
        elif resnet == 'resnet50':
            rn: models.ResNet = models.resnet50(pretrained=pretrained)
        elif resnet == 'resnet101':
            rn: models.ResNet = models.resnet101(pretrained=pretrained)
        elif resnet == 'resnet152':
            rn: models.ResNet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Specified Resnet model not available, choose one of:\n"
                             "['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']")

        self.features = nn.Sequential(*list(rn.children())[:-1])  # Exclude the original fc layer

        # Run some dummy data to get dimensions
        tmp = torch.ones((10, 3, height, width))
        tmp2 = self.features(tmp)
        print(tmp2.shape)

        in_features = tmp2.shape[1]

        # self.final = nn.Sequential(nn.Linear(self.h2, num_clust),
        #                            nn.BatchNorm1d(num_clust, momentum=0.01))
        self.final = nn.Linear(in_features, num_clust)
        
        if activation == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')

        # self.resnet = rn

    def forward(self, x):
        
        x = self.embed_layers(x)
        x = self.act(x)

        return x
    
    def embed_layers(self, x):
        # Expanding to 3 channel image
        x = x.expand(-1, 3, -1, -1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)

        return x
