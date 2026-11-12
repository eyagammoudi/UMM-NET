import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        # self.bn1 = nn.InstanceNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(inplace=True)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, out_channel, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, out_channel, act)
        layer2 = LUConv(out_channel, out_channel, act)
    else:
        layer1 = LUConv(in_channel, in_channel,act)
        layer2 = LUConv(in_channel, out_channel,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel, out_channle, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, out_channle,act)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x, max_pool=True):
        
        if max_pool:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        else :
            out = self.ops(x)
            out_before_pool = out

        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans,act):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(2*outChans, outChans, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)

        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)

    def forward(self, x):
        out = self.final_conv(x)
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channels, strides=(64, 128, 256, 512), n_class=1, act='relu'):
        super(UNet3D, self).__init__()

        self.downs = nn.ModuleList([])
        for i in range(len(strides)):
            if i == 0:
                self.downs.append(DownTransition(in_channels, strides[0], act))
            else :
                self.downs.append(DownTransition(strides[i-1], strides[i], act))

        self.ups = nn.ModuleList([])
        for j in range(len(strides)-1, -1, -1):
            if j == 0:
                self.ups.append(OutputTransition(strides[0], n_class))
            else :
                self.ups.append(UpTransition(strides[j], strides[j - 1], act))

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            if i == len(self.downs) - 1:
                x, _ = d(x, max_pool=False)
            else :
                x, skip = d(x)
                skips.append(skip)   
        skips.reverse()
        for i, u in enumerate(self.ups):
            if isinstance(u, OutputTransition):
                x = u(x)
            else :
                x = u(x, skips[i])
        
        return x 
        