import time
import copy
import math

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Function
from net.U2net import *
from net.ASPPR import *

from torchsummary import summary


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs=None):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),   # Add the last layer
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class Classifier2(nn.Module):
    """Classifier model for Hyptension Classification."""

    def __init__(self, configs=None):
        super(Classifier2, self).__init__()

        self.layer1 =  nn.Linear(448, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64+5, 2)   # Add the last layer

    def forward(self, input, hemo):
        hid = self.layer1(input)
        hid = self.layer2(hid)
        out = self.layer3(torch.concat((hid, hemo), dim=1))
        return out


class Pretrain_V5(nn.Module):
    def __init__(self):
        super(Pretrain_V5, self).__init__()

        self.commonFE = TF_ASPPR_SE(1, 1)
        self.encoder = U2NetEncoder(in_ch=16)
        self.regressor = U2NetDecoder(out_ch=6)
        self.gap = nn.AdaptiveAvgPool1d(1)  # global adaptive avg pool
        self.classifier = Classifier2()
        self.domain_discriminator = Discriminator()

    def forward(self, x):
        comm_feats = self.commonFE(x)
        hx1, hx2, hx3, hx4, hx5 = self.encoder(comm_feats)
        side = self.regressor(hx1, hx2, hx3, hx4, hx5)

        code = self.gap(hx5).squeeze(-1)
        avg_reg_out = side.mean(dim=2)
        hemo = avg_reg_out[:, -5:]

        class_output = self.classifier(code, hemo)

        reversed_features = grad_reverse(code, alpha=0.0)
        domain_output = self.domain_discriminator(reversed_features)

        return class_output, side, domain_output, code


class Adapt_V5(nn.Module):
    def __init__(self, num_of_source=2, pretrain_model=None):
        super(Adapt_V5, self).__init__()
        self.num_of_source = num_of_source
        self.commonFE = TF_ASPPR_SE(1, 1)  # ch x 16
        self.DIFE = U2NetEncoder(in_ch=16)  # domain invariant feature extractor
        self.classifier = Classifier2()
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Load ASPPR_SE from pretrain_model if provided
        if pretrain_model is not None:
            self.commonFE.load_state_dict(pretrain_model.commonFE.state_dict())
            dsfe = U2NetEncoder(in_ch=16)
            dsfe.load_state_dict(pretrain_model.encoder.state_dict())
            self.DIFE.load_state_dict(pretrain_model.encoder.state_dict())
            dsr = U2NetDecoder(out_ch=6)
            dsr.load_state_dict(pretrain_model.regressor.state_dict())

            for i in range(num_of_source):
                setattr(self, f'DSFE{i}', dsfe)
                setattr(self, f'DSR{i}', dsr)  # domain specific regressor(DSR)
                setattr(self, f'DSD{i}', Discriminator())  # domain specific discriminator(DSD)

        else:
            for i in range(num_of_source):
                setattr(self, f'DSFE{i}', U2NetEncoder(in_ch=16))  # domain specific feature extractor(DSFE)
                setattr(self, f'DSR{i}', U2NetDecoder(out_ch=6))  # domain specific regressor(DSR)
                setattr(self, f'DSD{i}', Discriminator())  # domain specific discriminator(DSD)

    def forward(self, data_src, data_tgt, branch_mark=0):
        """When data src and data tgt are the same, both are target test data"""
        data_tgt_DSFE_feats = []
        data_tgt_DSFE_code = []
        data_tgt_DSR_preds = []
        data_tgt_DSD_pred = []
        data_src_DSD_pred = []

        data_src_comm_feats = self.commonFE(data_src)  # common feature extractor output
        data_tgt_comm_feats = self.commonFE(data_tgt)  # common feature extractor output

        # data_src_code = self.gap(data_src_code).squeeze()
        # data_tgt_code = self.gap(data_tgt_code).squeeze()

        # Calculate target domain branch feature output
        for i in range(self.num_of_source):
            DSFE_i = getattr(self, f'DSFE{i}')
            data_tgt_DSFE_i = DSFE_i(data_tgt_comm_feats)
            data_tgt_DSFE_feats.append(data_tgt_DSFE_i)
            data_tgt_DSFE_code.append(self.gap(data_tgt_DSFE_i[-1]).squeeze(-1))

        # Calculate target domain branch regression prediction results
        for i in range(self.num_of_source):
            DSR_i = getattr(self, f'DSR{i}')
            data_tgt_DSR_i = DSR_i(*data_tgt_DSFE_feats[i])
            data_tgt_DSR_preds.append(data_tgt_DSR_i)

        # Calculate source domain branch feature output
        DSFE_branch = getattr(self, f'DSFE{branch_mark}')
        data_src_DSFE_feats = DSFE_branch(data_src_comm_feats)
        data_src_DSFE_code = self.gap(data_src_DSFE_feats[-1]).squeeze(-1)

        # Calculate source domain branch regression prediction results
        DSR_branch = getattr(self, f'DSR{branch_mark}')
        data_src_DSR_pred = DSR_branch(*data_src_DSFE_feats)

        # Calculate source and target domain discriminator prediction results
        DSD_branch = getattr(self, f'DSD{branch_mark}')
        data_tgt_DSD_feat = grad_reverse(data_tgt_DSFE_code[branch_mark], alpha=0)
        data_tgt_DSD_pred = DSD_branch(data_tgt_DSD_feat)

        data_src_DSD_feat = grad_reverse(data_src_DSFE_code, alpha=0)
        data_src_DSD_pred = DSD_branch(data_src_DSD_feat)

        # Calculate domain-invariant feature extractor's features
        data_src_DIFE_feat = self.DIFE(data_src_comm_feats)
        data_src_DIFE_code = self.gap(data_src_DIFE_feat[-1]).squeeze(-1)

        # Calculate the final classification result
        avg_tgt_DSR_pred = torch.stack(data_tgt_DSR_preds).mean(dim=0).mean(dim=2)  # The prediction results of multiple DSR branch averages are calculated here
        hemo = avg_tgt_DSR_pred[:, -5:]

        class_pred = self.classifier(data_src_DIFE_code, hemo)

        return data_tgt_DSFE_code, data_tgt_DSR_preds, data_src_DSFE_code, data_src_DSR_pred, data_tgt_DSD_pred, data_src_DSD_pred, class_pred
