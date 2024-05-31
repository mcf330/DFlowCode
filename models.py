import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import modules
import math

LRELU_SLOPE = 0.1


class SpectrogramUpsampler(nn.Module):
    def __init__(self, up_rates=[4, 4, 4, 4], condition_channel=80):
        super(SpectrogramUpsampler, self).__init__()
        self.up_convs = nn.ModuleList()
        for i in range(len(up_rates)):
            self.up_convs.append(weight_norm(
                nn.ConvTranspose1d(condition_channel, condition_channel, 2 * up_rates[i], stride=up_rates[i],
                                   padding=up_rates[i] // 2)))

    def forward(self, c):
        c_list = [c]
        for conv in self.up_convs:
            c = conv(c)
            c_list += [c]
            c = F.leaky_relu(c, LRELU_SLOPE)
        return c_list

    def remove_weight_norm(self):
        for conv in self.up_convs:
            remove_weight_norm(conv)


class ARFlowBlock(nn.Module):
    def __init__(self, dim, blocklayers=8, padding_left=True, out_h=True, condition_channel=80):
        super(ARFlowBlock, self).__init__()
        self.start = weight_norm(nn.Conv1d(1, dim, 1))
        self.padding_left = padding_left
        self.out_h = out_h
        self.wn_convs = nn.ModuleList()
        for layer in range(blocklayers):
            dilation = 2 ** layer
            self.wn_convs.append(
                modules.ResidualWaveNet(dim, 2 * dim, 2, condition_channel=condition_channel, dilation=dilation,
                                        padding_left=padding_left))
        if out_h:
            self.out = weight_norm(nn.Conv1d(dim, dim, 1))
        end = nn.Conv1d(dim, 2, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

    def forward(self, x, c=None, h=None):
        x_o = x
        if self.padding_left:
            x = torch.cat([x[:, :, 0:1], x[:, :, :-1]], -1)
        else:
            x = torch.cat([x[:, :, 1:], x[:, :, -1:]], -1)
        x = self.start(x)
        if h is not None:
            x = x + h
        skips = 0
        for f in self.wn_convs:
            x, s = f(x, c)
            skips += s
        skips *= math.sqrt(1.0 / len(self.wn_convs))
        x = F.leaky_relu(skips, LRELU_SLOPE)
        if self.out_h:
            h = self.out(x)
        x = self.end(x)
        mu, logs = x.split(x.size(1) // 2, dim=1)
        x = x_o * torch.exp(logs) + mu
        return logs, x, h

    def remove_weight_norm(self):
        remove_weight_norm(self.start)
        for f in self.wn_convs:
            f.remove_weight_norm()
        if self.out_h:
            remove_weight_norm(self.out)


class AuxiliaryFlow(nn.Module):
    def __init__(self, dim, layers=8, condition_channel=80):
        super(AuxiliaryFlow, self).__init__()
        self.arflowsleft = nn.ModuleList()
        self.arflowsright = nn.ModuleList()
        layers_half = layers // 2
        for i in range(layers_half):
            if i < layers_half - 1:
                out_h = True
            else:
                out_h = False
            self.arflowsleft.append(
                ARFlowBlock(dim, layers, condition_channel=condition_channel, padding_left=True, out_h=out_h))
            self.arflowsright.append(
                ARFlowBlock(dim, layers, condition_channel=condition_channel, padding_left=False, out_h=out_h))

    def forward(self, x, c):
        logs_all = 0
        h = None
        for arflow in self.arflowsleft:
            logs, x, h = arflow(x, c, h)
            logs_all += logs
        h = None
        for arflow in self.arflowsright:
            logs, x, h = arflow(x, c, h)
            logs_all += logs
        z = x
        return logs_all, z

    def remove_weight_norm(self):
        for arflow in self.arflowsleft:
            arflow.remove_weight_norm()
        for arflow in self.arflowsright:
            arflow.remove_weight_norm()


class PrimaryFlow(nn.Module):
    def __init__(self, n_flows, squeeze_ch=4, condition_channel=80, unet_ch=64, up_rates=[4, 4, 4]):
        super(PrimaryFlow, self).__init__()
        self.squeeze_ch = squeeze_ch
        self.n_flows = n_flows
        self.flowblocks = nn.ModuleList()
        for i in range(n_flows):
            self.flowblocks.append(
                modules.AffineLayer(squeeze_ch // 2, unet_ch, up_rates=up_rates, condition_channel=condition_channel))

    def forward(self, x, c_list):
        x = x.squeeze(1).unfold(1, self.squeeze_ch, self.squeeze_ch).permute(0, 2, 1)  # squeeze
        for i in range(self.n_flows):
            x = torch.flip(x, [1])
            n_half = self.squeeze_ch // 2
            x_0 = x[:, :n_half, :]
            x_1 = x[:, n_half:, :]
            bias = self.flowblocks[i](x_0, c_list)
            x_1 = x_1 + bias
            x = torch.cat([x_0, x_1], 1)
        return x

    def backward(self, x, c_list):
        x = x.squeeze(1).unfold(1, self.squeeze_ch, self.squeeze_ch).permute(0, 2, 1)
        for k in reversed(range(self.n_flows)):
            n_half = self.squeeze_ch // 2
            x_0 = x[:, :n_half, :]
            x_1 = x[:, n_half:, :]
            bias = self.flowblocks[k](x_0, c_list)
            x_1 = x_1 - bias
            x = torch.cat([x_0, x_1], 1)
            x = torch.flip(x, [1])
        x = x.transpose(1, 2).contiguous().view(x.size(0), 1, -1)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flowblocks[i].remove_weight_norm()


class Decoder(nn.Module):
    def __init__(self, unet_dim, n_unets, condition_channel=80, up_rates=[4, 4, 4, 4]):
        super(Decoder, self).__init__()
        self.start = weight_norm(nn.Conv1d(1, unet_dim, 1))
        self.end = nn.Conv1d(unet_dim, 1, 11, padding=11 // 2)
        self.unet_list = nn.ModuleList()
        for i in range(n_unets):
            self.unet_list.append(modules.UNet(unet_dim, up_rates=up_rates, condition_channel=condition_channel))

    def forward(self, x, c_list):
        x = self.start(x)
        for unet in self.unet_list:
            x = unet(x, c_list)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.end(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.start)
        for unet in self.unet_list:
            unet.remove_weight_norm()


class DFlow(nn.Module):
    def __init__(self, auxiliary_flow_dim=32, auxiliary_layers=8, primary_flow_dim=64, primary_flow_layers=12,
                 decoder_dim=32, decoder_layers=3, condition_channel=80, up_rates=[4, 4, 4, 4]):
        super(DFlow, self).__init__()
        self.condition_upsampling = SpectrogramUpsampler(up_rates=up_rates, condition_channel=condition_channel)
        self.auxiliary_flow = AuxiliaryFlow(auxiliary_flow_dim, layers=auxiliary_layers,
                                            condition_channel=condition_channel)
        self.primary_flow = PrimaryFlow(primary_flow_layers, condition_channel=condition_channel,
                                        unet_ch=primary_flow_dim)
        self.decoder = Decoder(decoder_dim, decoder_layers, condition_channel=condition_channel)

    def forward(self, x, c):
        c_list = self.condition_upsampling(c)
        logs_auxiliary_flow, z_l = self.auxiliary_flow(x, c_list[-1])
        c_list.reverse()
        z_p = self.primary_flow(z_l, c_list[1:])
        x_hat = self.decoder(z_l, c_list)
        return z_p, logs_auxiliary_flow, x_hat

    def infer(self, c, t=1.0):
        c_list = self.condition_upsampling(c)
        z_p = torch.randn(c_list[-1].size(0), 1, c_list[-1].size(-1)).type_as(c) * t
        c_list.reverse()
        z_l = self.primary_flow.backward(z_p, c_list[1:])
        x_hat = self.decoder(z_l, c_list)
        return x_hat

    def remove_weight_norm(self):
        self.condition_upsampling.remove_weight_norm()
        self.auxiliary_flow.remove_weight_norm()
        self.primary_flow.remove_weight_norm()
        self.decoder.remove_weight_norm()
