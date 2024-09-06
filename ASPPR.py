import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
def _get_act_func(act_func, in_channels):
    if act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'leaky_relu':
        # return nn.LeakyReLU(0.05)
        return nn.LeakyReLU(0.01)
    elif act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'prelu':
        return nn.PReLU(init=0.05)
    elif act_func == 'cprelu':
        return nn.PReLU(num_parameters=in_channels, init=0.05)
    elif act_func == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.cat([r, p], -1), out_ft


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias=True, padding_mode='zeros', use_bn=True, use_act=True, act='relu'):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act

        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bn = nn.BatchNorm1d(out_channel)
        self.acti = _get_act_func(act, in_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.acti(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, p=0.0):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.bn1 = nn.BatchNorm1d(in_channel)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=int((kernel_size - stride) / 2), dilation=1, groups=1)

        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p)

        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride=1, padding=int((kernel_size - 1) / 2), dilation=1, groups=1)

        self.avg_pool = nn.AvgPool1d(kernel_size=stride) if stride > 1 else None

    def forward(self, x):
        net = self.conv1(self.relu1(self.bn1(x)))
        net = self.conv2(self.relu2(self.bn2(net)))

        if self.stride > 1:
            x = self.avg_pool(x)

        return net + x

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilations=(1, 6, 12, 18), use_bn=True, use_act=True, act_func='relu'):
        super(ASPP, self).__init__()

        self.num_scale = len(dilations)

        self.convs = nn.ModuleList()
        for dilation in dilations:
            padding = dilation * (kernel_size - 1) // 2
            self.convs.append(ConvBlock(in_channel, out_channel, kernel_size, stride=1, padding=padding, dilation=dilation, groups=1, use_bn=use_bn, use_act=use_act, act=act_func))

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        return torch.cat(feats, dim=1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(np.ceil(c2 * e))  # hidden channels
        self.cv1 = nn.Conv1d(c1, c_, 1, 1)
        self.cv2 = nn.Conv1d(c_, c2, 3, 1, padding=1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class ASPPR_SE(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16, aspp_bn=True, aspp_act=True, p=0.0, dilations=(1, 6, 12, 18), act_func='relu'):
        super(ASPPR_SE, self).__init__()

        self.dila_num = len(dilations)

        self.aspp1 = ASPP(in_ch, out_ch, kernel_size=3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer1 = SELayer(out_ch * self.dila_num, reduction=reduction)
        self.residual1 = ResidualBlock(out_ch * self.dila_num, out_ch * self.dila_num, kernel_size=3, stride=1, p=p)

        self.aspp2 = ASPP(out_ch * self.dila_num, out_ch, kernel_size=3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer2 = SELayer(out_ch * self.dila_num, reduction=reduction)
        self.residual2 = ResidualBlock(out_ch * self.dila_num, out_ch * self.dila_num, kernel_size=3, stride=1, p=p)

        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # Add a Bottleneck layer to reduce the number of channels
        self.bottleneck = Bottleneck(out_ch * self.dila_num, out_ch * self.dila_num // 4, shortcut=False)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        y = self.aspp1(x)
        y = self.se_layer1(y)
        y = self.residual1(y)
        # y = self.aspp2(y)
        # y = self.se_layer2(y)
        # y = self.residual2(y)
        bt = self.bottleneck(y)
        out = self.bn(bt)
        # out = self.gap(out)
        return out.squeeze()

class TF_ASPPR_SE(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16, aspp_bn=True, aspp_act=True, p=0.0, dilations=(1, 6, 12, 18), act_func='relu'):
        super(TF_ASPPR_SE, self).__init__()

        self.dila_num = len(dilations)

        self.freq_feature = SpectralConv1d(in_channels = 1, out_channels= 1, modes1=250)
        self.time_feature = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        self.bn_freq = nn.BatchNorm1d(250*2)

        self.aspp1 = ASPP(in_ch, out_ch, kernel_size=3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer1 = SELayer(out_ch * self.dila_num, reduction=1)
        self.residual1 = ResidualBlock(out_ch * self.dila_num, out_ch * self.dila_num, kernel_size=3, stride=1, p=p)

        self.aspp2 = ASPP(out_ch * self.dila_num, out_ch * self.dila_num, kernel_size=3, dilations=dilations, use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer2 = SELayer(out_ch * self.dila_num *4, reduction=4)
        self.residual2 = ResidualBlock(out_ch * self.dila_num * 4, out_ch * self.dila_num *4, kernel_size=3, stride=1, p=p)

        self.bn = nn.BatchNorm1d(out_ch * self.dila_num * 4)
        self.relu = nn.ReLU()

        # Add a Bottleneck layer to reduce the number of channels
        self.bottleneck = Bottleneck(out_ch * self.dila_num, out_ch * self.dila_num // 4, shortcut=False)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # ffeats, _ = self.freq_feature(x)
        # tfeats = self.time_feature(x)
        # ffeats2 = F.relu(self.bn_freq(ffeats.squeeze()))
        #
        # tf_feats = torch.cat([tfeats, ffeats2.unsqueeze(dim=1)], dim=-1) # 可以随时取消不用，下面一行输入直接替换成x即可

        y = self.aspp1(x)
        y = self.se_layer1(y)
        y = self.residual1(y)
        y = self.aspp2(y)
        y = self.se_layer2(y)
        y = self.residual2(y)
        # bt = self.bottleneck(y)
        out = self.bn(y)
        # out = self.gap(out)
        return out
