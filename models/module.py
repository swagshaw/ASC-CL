import torch.nn as nn
import torch.nn.functional as F
from utils.models import init_layer, init_bn

"""addapted from https://github.com/roman-vygon/BCResNet"""


class ConvBlock3x3(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock3x3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class SubSpectralNorm(nn.Module):
    def __init__(self, C, S, eps=1e-5):
        super(SubSpectralNorm, self).__init__()
        self.S = S
        self.eps = eps
        self.bn = nn.BatchNorm2d(C * S)

    def forward(self, x):
        N, C, T, F = x.size()
        # Changed view to reshape for quantisatoin error fix
        x = x.reshape(N, C * self.S, T, F // self.S)
        x = self.bn(x)
        return x.reshape(N, C, T, F)


class BroadcastedBlock(nn.Module):
    def __init__(
            self,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ):
        super(BroadcastedBlock, self).__init__()
        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      dilation=dilation,
                                      stride=stride, bias=False)
        self.ssn1 = SubSpectralNorm(planes, 4)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x

        # f2
        ##########################
        out = self.freq_dw_conv(x)
        out = self.ssn1(out)
        ##########################

        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        ############################
        out = self.temp_dw_conv(out)
        out = self.bn(out)
        out = self.swish(out)
        out = self.conv1x1(out)
        out = self.channel_drop(out)
        ############################

        out = out + identity + auxilary
        out = self.relu(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(
            self,
            inplanes: int,
            planes: int,
            dilation=1,
            stride=1,
            temp_pad=(0, 1),
    ):
        super(TransitionBlock, self).__init__()

        self.freq_dw_conv = nn.Conv2d(planes, planes, kernel_size=(3, 1), padding=(1, 0), groups=planes,
                                      stride=stride,
                                      dilation=dilation, bias=False)
        self.ssn = SubSpectralNorm(planes, 4)
        # self.ssn = nn.BatchNorm2d(planes)
        self.temp_dw_conv = nn.Conv2d(planes, planes, kernel_size=(1, 3), padding=temp_pad, groups=planes,
                                      dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.channel_drop = nn.Dropout2d(p=0.1)
        self.swish = nn.SiLU()
        self.conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=(1, 1), bias=False)
        self.flag = False

    def forward(self, x):
        # f2
        #############################
        out = self.conv1x1_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.freq_dw_conv(out)
        out = self.ssn(out)
        #############################
        auxilary = out
        out = out.mean(2, keepdim=True)  # frequency average pooling

        # f1
        #############################
        out = self.temp_dw_conv(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv1x1_2(out)
        out = self.channel_drop(out)
        #############################

        out = auxilary + out
        out = self.relu(out)

        return out
