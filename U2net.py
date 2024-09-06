import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ASPPR import ASPPR_SE
from torchsummary import summary

class ConvBN(nn.Module):

    def __init__(self, in_ch, out_ch, kernal_size=3, strides=1, padding=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch,
                              kernel_size=kernal_size, stride=strides,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='linear', align_corners=True)
    return src


class UUnit(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(UUnit, self).__init__()

        self.Conv_bn_in = ConvBN(in_ch, out_ch)

        self.Conv_bn1 = ConvBN(out_ch, mid_ch)
        self.pool1 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.Conv_bn2 = ConvBN(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.Conv_bn3 = ConvBN(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.Conv_bn4 = ConvBN(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.Conv_bn5 = ConvBN(mid_ch, mid_ch)

        self.Conv_bn6 = ConvBN(mid_ch, mid_ch)

        self.Conv_bn5d = ConvBN(mid_ch * 2, mid_ch)
        self.Conv_bn4d = ConvBN(mid_ch * 2, mid_ch)
        self.Conv_bn3d = ConvBN(mid_ch * 2, mid_ch)
        self.Conv_bn2d = ConvBN(mid_ch * 2, mid_ch)
        self.Conv_bn1d = ConvBN(mid_ch * 2, out_ch)

    def forward(self, x):
        hx = x
        hxin = self.Conv_bn_in(hx)

        hx1 = self.Conv_bn1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.Conv_bn2(hx)
        hx = self.pool2(hx2)

        hx3 = self.Conv_bn3(hx)
        hx = self.pool3(hx3)

        hx4 = self.Conv_bn4(hx)
        hx = self.pool4(hx4)

        hx5 = self.Conv_bn5(hx)

        hx6 = self.Conv_bn6(hx5)

        hx5d = self.Conv_bn4d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5, hx4)

        hx4d = self.Conv_bn4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.Conv_bn3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.Conv_bn2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.Conv_bn1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

class U2NetEncoder(nn.Module):
    def __init__(self, in_ch):
        super(U2NetEncoder, self).__init__()

        self.stage1 = UUnit(16, 14, 28)
        self.pool12 = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.stage2 = UUnit(28, 14, 56)
        self.pool23 = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.stage3 = UUnit(56, 28, 112)
        self.pool34 = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.stage4 = UUnit(112, 56, 224)
        self.pool45 = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        self.stage5 = UUnit(224, 112, 448)

    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)

        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)

        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)

        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)

        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)

        return hx1, hx2, hx3, hx4, hx5


class U2NetDecoder(nn.Module):
    def __init__(self, out_ch):
        super(U2NetDecoder, self).__init__()

        # decoder
        self.stage4d = UUnit(672, 336, 168)
        self.stage3d = UUnit(280, 140, 70)
        self.stage2d = UUnit(126, 63, 32)
        self.stage1d = UUnit(60, 30, 15)

        self.side1 = nn.Conv1d(15, out_ch, 3, padding=1)

    def forward(self, hx1, hx2, hx3, hx4, hx5):
        hx5up = _upsample_like(hx5, hx4)

        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        return self.side1(hx1d)
