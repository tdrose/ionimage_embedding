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

        if activation == 'softmax':
            self.fc = nn.Sequential(nn.Linear(in_features, num_clust), 
                                    nn.Softmax(dim=1)) # type: ignore
        elif activation == 'sigmoid':
            self.fc = nn.Sequential(nn.Linear(in_features, num_clust), 
                                    nn.Sigmoid()) # type: ignore
        elif activation == 'relu':
            self.fc = nn.Sequential(nn.Linear(in_features, num_clust), 
                                    nn.ReLU()) # type: ignore
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')

        self.resnet = rn

    def forward(self, x):
        # Expanding to 3 channel image
        x = x.expand(-1, 3, -1, -1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def embed_layers(self, x):
        # Currently just a dummy for forward pass
        return self.forward(x)