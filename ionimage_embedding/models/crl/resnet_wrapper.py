import torch
import torchvision.models as models


class ResNetWrapper(torch.nn.Module):

    def __init__(self):

        super(ResNetWrapper, self).__init__()
