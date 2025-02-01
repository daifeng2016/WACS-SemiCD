from .backbones.resnet2 import resnet50 as resnet

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class DeepLabV3Plus(nn.Module):
    def __init__(self, backname='resnet50',dilations=[6,12,18],num_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.is_corr = True
        self.backbone = resnet(pretrained=True,replace_stride_with_dilation=[False,False,True])


        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, dilations)

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, num_classes, 1, bias=True)
        if self.is_corr:
            self.corr = Corr(nclass=num_classes)
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x1, x2, need_fp=False,use_corr=False):
        dict_return = {}
        h, w = x1.shape[-2:]  # [256,256]

        feats1 = self.backbone.base_forward(x1)  # [2,256,64,64], [2,512,32,32], [2,1024,16,16], [2,2048,16,16]
        c11, c14 = feats1[0], feats1[-1]

        feats2 = self.backbone.base_forward(x2)
        c21, c24 = feats2[0], feats2[-1]

        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs()

        # if need_fp:
        #     outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
        #                         torch.cat((c4, nn.Dropout2d(0.5)(c4))))  # [4,2,64,64]
        #     outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)  # [4,2,256,256]
        #     out, out_fp = outs.chunk(2)  # [2,2,256,256]
        #
        #     return out, out_fp
        if need_fp:
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))#[8,256,64,64]
            outs = self.classifier(feats_decode)#[8,19,64,64]
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)#[8,19,256,256]
            out, out_fp = outs.chunk(2)#[4,19,256,256]
            if use_corr:
                proj_feats = self.proj(c4)#[4,256,32,32]
                corr_out = self.corr(proj_feats, out)#[4,19,32,32]
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)##[4,19,256,256]
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return





        # out = self._decode(c1, c4)
        # out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        feats_decode = self._decode(c1, c4)  # [2,256,64,64]
        out = self.classifier(feats_decode)  # [2,19,64,64]
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)  # [2,19,256,256]
        if use_corr:
            proj_feats = self.proj(c4)  # [2,256,32,32]
            corr_out = self.corr(proj_feats, out)  # [2,19,32,32]
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)  # [2,19,256,256]
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return





    # def _decode(self, c1, c4):
    #     c4 = self.head(c4)
    #     c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
    #
    #     c1 = self.reduce(c1)
    #
    #     feature = torch.cat([c1, c4], dim=1)
    #     feature = self.fuse(feature)
    #
    #     out = self.classifier(feature)
    #
    #     return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)#[2,256,32,32]
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)#[2,256,64,64]

        c1 = self.reduce(c1)#[2,48,64,64]

        feature = torch.cat([c1, c4], dim=1)#[2,304,64,64]
        feature = self.fuse(feature)#[2,256,64,64]

        return feature



def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):#[4,256,32,32],[4,19,256,256]
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))#32,32
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)#[4,19,32,32]
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)#[4,256,32,32]
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)') #eq.(7) [4,19,1024]
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')#[4,19,1024]
        out_temp = rearrange(out, 'n c h w -> n c (h w)')#[4,19,1024]
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())#eq.(7)
        corr_map = F.softmax(corr_map, dim=-1)#[4,1024,1024] eq.(8)
        out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)#[4,19,32,32], eq.(8)
        return out