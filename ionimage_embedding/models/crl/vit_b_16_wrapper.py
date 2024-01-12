import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights

from typing import Literal, Optional

class VitB16Wrapper(torch.nn.Module):
    def __init__(self, 
                 num_clust: int, 
                 height: int, width: int,
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 pretrained: Optional[Literal['IMAGENET1K_V1', 'IMAGENET1K_SWAG_E2E_V1', 
                                              'IMAGENET1K_SWAG_LINEAR_V1']] = None,
                 ):
        
        super(VitB16Wrapper, self).__init__()
        
        if height != width:
            raise ValueError('VitB16Wrapper requires square images.')
        
        if pretrained is None:
            self.vit_b_16 = vit_b_16(image_size=height)
        elif pretrained == 'IMAGENET1K_V1':
            self.vit_b_16 = vit_b_16(pretrained=ViT_B_16_Weights.IMAGENET1K_V1)
        elif pretrained == 'IMAGENET1K_SWAG_E2E_V1':
            self.vit_b_16 = vit_b_16(pretrained=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        elif pretrained == 'IMAGENET1K_SWAG_LINEAR_V1':
            self.vit_b_16 = vit_b_16(pretrained=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        else:
            raise ValueError('Pretrained weights not available. '
                             'Use IMAGENET1K_V1, IMAGENET1K_SWAG_E2E_V1, or '
                             'IMAGENET1K_SWAG_LINEAR_V1.')

        if activation == 'softmax':
            self.final = nn.Sequential(nn.Linear(self.vit_b_16.num_classes, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.Softmax(dim=1))
        elif activation == 'sigmoid':
            self.final = nn.Sequential(nn.Linear(self.vit_b_16.num_classes, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.Sigmoid())
        elif activation == 'relu':
            self.final = nn.Sequential(nn.Linear(self.vit_b_16.num_classes, num_clust),
                                    nn.BatchNorm1d(num_clust, momentum=0.01),
                                    nn.ReLU())
        else:
            raise ValueError('Activation function not available. Use softmax, relu, or sigmoid.')

    def forward(self, x):
        # Expanding to 3 channel image
        x = x.expand(-1, 3, -1, -1)
        x = self.vit_b_16(x)
        x = self.final(x)
        return x

    def embed_layers(self, x):
        # Currently just a dummy for forward pass
        return self.forward(x)
