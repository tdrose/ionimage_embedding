import math


def conv2d_hout(height, padding, dilation, kernel_size, stride):
    tmp = math.floor(((height + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0]) + 1)
    return 1 if tmp < 1 else tmp


def conv2d_wout(width, padding, dilation, kernel_size, stride):
    tmp = math.floor(((width + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1) / stride[1]) + 1)
    return 1 if tmp < 1 else tmp


def conv2d_hwout(height, width, padding, dilation, kernel_size, stride):
    h = math.floor(((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
    h = 1 if h < 1 else h
    w = math.floor(((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
    w = 1 if w < 1 else w

    return h, w