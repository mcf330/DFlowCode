import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
import math

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)


class UNet(nn.Module):
    def __init__(self, start_ch=32, up_rates=[4, 4, 4, 4], condition_channel=80):
        super(UNet, self).__init__()
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.mel_mappings = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.donwsamplings = nn.ModuleList()
        ch = start_ch
        self.blocksize = len(up_rates)
        # donwsample
        self.mel_mappings.append(weight_norm(nn.Conv1d(condition_channel, ch, 1)))
        for i in range(self.blocksize):
            scale = up_rates[i]
            self.donwsamplings.append(weight_norm(nn.Conv1d(ch, ch * 2, scale * 2, stride=scale, padding=scale // 2)))
            self.mel_mappings.append(weight_norm(nn.Conv1d(condition_channel, ch * 2, 1)))
            self.down_convs.append(ResBlock(ch * 2))
            ch = ch * 2
        # upsample
        self.up_convs.append(ResBlock(ch))
        for i in range(self.blocksize):
            scale = up_rates[self.blocksize - i - 1]
            self.upsamplings.append(
                weight_norm(nn.ConvTranspose1d(ch, ch // 2, scale * 2, stride=scale, padding=scale // 2)))
            self.up_convs.append(ResBlock(ch // 2))
            ch = ch // 2

    def forward(self, x, c_list):
        c_list_mappings = []
        for i in range(len(c_list)):
            c_list_mappings += [self.mel_mappings[i](c_list[i])]
        mutiscale_residual = [x]
        x = x + c_list_mappings[0]
        for i in range(self.blocksize):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.donwsamplings[i](x)
            x = self.down_convs[i](x + c_list_mappings[i + 1])
            mutiscale_residual += [x]
        x = self.up_convs[0](x + c_list_mappings[self.blocksize] + mutiscale_residual[self.blocksize])
        for i in range(self.blocksize):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.upsamplings[i](x)
            x = self.up_convs[i + 1](
                x + c_list_mappings[self.blocksize - i - 1] + mutiscale_residual[self.blocksize - i - 1])
        return x

    def remove_weight_norm(self):
        for convblock in self.down_convs:
            convblock.remove_weight_norm()
        for convblock in self.up_convs:
            convblock.remove_weight_norm()
        for conv in self.mel_mappings:
            remove_weight_norm(conv)
        for conv in self.upsamplings:
            remove_weight_norm(conv)
        for conv in self.donwsamplings:
            remove_weight_norm(conv)


class AffineLayer(nn.Module):
    def __init__(self, input_ch, unet_ch, up_rates=[4, 4, 4], condition_channel=80):
        super(AffineLayer, self).__init__()
        self.start = weight_norm(nn.Conv1d(input_ch, unet_ch, 1))
        self.unet = UNet(unet_ch, up_rates, condition_channel=condition_channel)
        self.end = nn.Conv1d(unet_ch, input_ch, 1)

    def forward(self, x, c_list):
        x = self.start(x)
        x = self.unet(x, c_list)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.end(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.start)
        self.unet.remove_weight_norm()


class ResidualWaveNet(nn.Module):
    def __init__(self, residual_ch, gate_ch, k, condition_channel=80, dilation=1, padding_left=True):
        super(ResidualWaveNet, self).__init__()
        self.conv = weight_norm(nn.Conv1d(residual_ch, gate_ch, k, padding=(k - 1) * dilation, dilation=dilation))
        if condition_channel > 0:
            self.conv_c = weight_norm(nn.Conv1d(condition_channel, gate_ch, 1))
        gate_out_channels = gate_ch // 2
        self.conv_out = weight_norm(nn.Conv1d(gate_out_channels, residual_ch, 1))
        self.conv_skip = weight_norm(nn.Conv1d(gate_out_channels, residual_ch, 1))
        self.padding_left = padding_left

    def forward(self, x, c=None):
        res = x
        x = self.conv(x)
        if self.padding_left:
            x = x[:, :, :res.size(-1)]
        else:
            x = x[:, :, -res.size(-1):]
        if c is not None:
            x = x + self.conv_c(c)
        a, b = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(a) * torch.sigmoid(b)
        s = self.conv_skip(x)
        x = self.conv_out(x)
        out = (x + res) * math.sqrt(0.5)
        return out, s

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)
        remove_weight_norm(self.conv_c)
        remove_weight_norm(self.conv_out)
        remove_weight_norm(self.conv_skip)
