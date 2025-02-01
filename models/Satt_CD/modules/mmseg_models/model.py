# %%
import yaml, math, os

# with open('config.yaml') as fh:
#     config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn.functional as F
import torch.nn as nn

#from .backbones.mscan import MSCAN
#from .decode_heads.ham_head import LightHamHead
from .decode_heads.aspp_head2 import ASPP_Head
import torchvision.models as models
from .backbones.resnet2 import resnet50 as resnet

class CNN(nn.Module):
    def __init__(self, dilated=False, pretrained_base=True, **kwargs):
        super(CNN, self).__init__()
        self.pretrained =models.resnet34(pretrained=True)
            #eval(cnn_name)(pretrained=pretrained_base, dilated=dilated, **kwargs)

    def forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        p1 = self.pretrained.layer1(x)
        p2 = self.pretrained.layer2(p1)
        p3 = self.pretrained.layer3(p2)
        p4 = self.pretrained.layer4(p3)

        return p1, p2, p3, p4
class SegNeXtCD(nn.Module):
    def __init__(self, num_classes=2, in_channnels=3, embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256,  dropout=0.0, drop_path=0.0,pretrained=False):
        super().__init__()
        # self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
        #                               nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        #directly use MSCAN backbone not work for CD, as no much multi-scale change are needed
        #self.backbone=MSCAN(in_chans=in_channnels, embed_dims=embed_dims,mlp_ratios=ffn_ratios,depths=depths)
        self.backbone=CNN()#self.head_dim = [64, 128, 256, 512]
        #self.backbone=resnet(pretrained=True,replace_stride_with_dilation=[False,False,True])
        # self.decode_head = LightHamHead(
        #     # type='LightHamHead',
        #     in_channels=[128, 256, 512],  # MSCAN:[64, 160, 256]
        #     in_index=[1, 2, 3],
        #     channels=512,
        #     ham_channels=512,
        #     ham_kwargs=dict(MD_R=16),
        #     dropout_ratio=0.1,
        #     num_classes=num_classes,
        #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        #     align_corners=False
        # )
        self.decode_head = LightHamHead(
            # type='LightHamHead',
            in_channels=[128, 256, 512],  # MSCAN:[64, 160, 256]
            in_index=[1, 2, 3],
            channels=512,
            ham_channels=512,
            ham_kwargs=dict(MD_R=16),
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            align_corners=False
        )

        low_channels = 32
        high_channels = 256

        # self.decode_head=ASPP_Head(
        #     low_channels=low_channels,
        #     high_channels=high_channels,
        #     reduce=4,
        #     low_reduce=32,
        # )



        self.pretrained=pretrained

        ###self.init_weights()#must not use init_weights after loading pretrained weight!!!

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, mean=0.0, std=0.02)
    #         if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, val=1.0)
    #             nn.init.constant_(m.bias, val=0.0)
    #         if isinstance(m, nn.Conv2d):
    #             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             fan_out //= m.groups
    #             nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out), mean=0)

    def forward(self, x1,x2):  # [2,3,256,256]

        # enc_feats = self.encoder(x)  # [2,32,64,64],[2,64,32,32],[2,460,16,16],[2,256,8,8]
        # dec_out = self.decoder(enc_feats)  # [2,256,32,32]
        # if self.pretrained:
        #     #model_path = r'D:\BaiduNetdiskDownload\Pretrained\segnext_tiny_512x512_ade_160k.pth'
        #     model_path = r'D:\BaiduNetdiskDownload\Pretrained\mscan_t.pth'
        #     state_dict = torch.load(model_path)
        #     for name, param in self.backbone.named_parameters():
        #         #name_full = 'backbone.' + name
        #         param.data.copy_(state_dict['state_dict'][name].data)


        enc_feats1=self.backbone(x1)
        enc_feats2=self.backbone(x2)
        #===========for resnet50=======================
        # enc_feats1 = self.backbone.base_forward(x1)
        # enc_feats2 = self.backbone.base_forward(x2)
        #================directly diff=====================================
        fuse_feats=[torch.abs(fea1-fea2) for fea1,fea2 in zip(enc_feats1,enc_feats2)]

        ##============for feature inteaction==================
        #1) 3D conv;2)linear cross-att;3) gcn;


        #=================decoder=======================
        dec_out=self.decode_head(fuse_feats)
        #dec_out=self.decode_head(enc_feats1,enc_feats2)

        dec_out = F.interpolate(dec_out, size=x1.size()[-2:], mode='bilinear',
                               align_corners=True)  # now its same as input [2,2,256,256]
        #  bilinear interpol was used originally
        return dec_out


def print_model_parm_nums(model):  # 得到模型参数总量

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
    return total / 1e6


if __name__ == '__main__':
    device = torch.device("cuda:0")
    input = torch.rand(2, 3, 256, 256)
    input = input.to(device)
    net = SegNext()
    print_model_parm_nums(net)

    model_path = r'D:\BaiduNetdiskDownload\Pretrained\segnext_tiny_512x512_ade_160k.pth'
    state_dict = torch.load(model_path)
    # for k in state_dict['state_dict'].keys():
    #     print(k)
    for name, parmas in net.named_parameters():
        print(name)
    #load for whole network
    #net.load_state_dict(state_dict['state_dict'], strict=False)
    #=====in fact, we only need the encoder==============
    encoder_dict=state_dict['state_dict']
    model_dict=net.state_dict()
    #method1
    # for k in list(encoder_dict.keys()):
    #     if not k.startswith('backbone'):
    #      del encoder_dict[k]
    #     #if k.k.startswith('backbone'):
    # net.backbone.load_state_dict(encoder_dict,strict=False)
    #method2
    for k in list(encoder_dict.keys()):
        if k.startswith('backbone'):
           model_dict[k].data.copy_(encoder_dict[k].data)



    net.to(device)
    output = net(input)

    # ==================for (out1,out2,out3) output===========================
    if isinstance(output, list) or isinstance(output, tuple):
        for i in range(len(output)):
            print(output[i].size())
    else:
        print(output.size())
