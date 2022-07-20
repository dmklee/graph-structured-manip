import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Norm_dict = {'identity' : nn.Identity,
             'batch' : nn.BatchNorm2d,
             'instance' : nn.InstanceNorm2d,
             'layer' : nn.LayerNorm,
           }

class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        return input.view(input.size(0), *self.shape)

class Broadcast2d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __repr__(self):
        return f'Broadcast2d'

    def forward(self, input):
        input = input.unsqueeze(2).unsqueeze(2)
        return input.expand(-1,-1,self.size, self.size)

def custom_initialize(network):
    for m in network.named_modules():
        if isinstance(m[1], nn.Conv2d):
            # nn.init.kaiming_normal_(m[1].weight.data)
            nn.init.xavier_normal_(m[1].weight.data)
        elif isinstance(m[1], nn.BatchNorm2d):
            m[1].weight.data.fill_(1)
            m[1].bias.data.zero_()
    return network

class DynFilter2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias=None):
        '''
        :x: torch tensor of shape (N,C_in,H,W)
        :weight: torch tensor of shape (N,C_out,C_in,kH,kW)
        :bias: torch tensor of shape (N,C_out)
        '''
        N,_,H,W = x.size()
        kH,kW = weight.shape[-2:]

        weight = weight.view(-1,*weight.shape[2:])
        x = x.view(1, -1, H, W)
        output = F.conv2d(x, weight=weight, padding=(kH//2,kW//2),
                                   groups=N)
        output = output.view(N, -1, H, W)
        if bias is not None:
            output += bias.view(N,-1,1,1)
        return output

def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    kernel_size = np.asarray((3, 3))
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )

def conv1x1(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "1x1 convolution with padding"
    kernel_size = np.asarray((1, 1))
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 bias=False):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation, bias=bias)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation, bias=bias)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def rand_perlin_2d(shape,
                   res,
                   fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]),
                                    torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * np.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave( d[1], 1)

    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return np.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]),
                                   torch.lerp(n01, n11, t[..., 0]), t[..., 1])
