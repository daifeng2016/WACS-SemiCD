# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:02:34 2021

@author: DELL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
#from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
#from .edge_utils import mask_to_onehot, onehot_to_binary_edges

class edgeBCE_Dice_loss(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, thresh=0.75):
        super(edgeBCE_Dice_loss, self).__init__()
        self.thresh = thresh

    def forward(self, pred, target, edge=None):
        DiceLoss_fn = DiceLoss(mode='binary',
                               from_logits=False)
        # 交叉熵
        BinaryCrossEntropy_fn = nn.BCELoss(reduction='none')
        edge_weight = 4.
        loss_bce = BinaryCrossEntropy_fn(pred, target)
        loss_dice = DiceLoss_fn(pred, target)
        if edge is not None:
            edge[edge == 0] = 1.
            edge[edge == 255] = edge_weight
            loss_bce *= edge
        # OHEM
        loss_bce_, ind = loss_bce.contiguous().view(-1).sort()
        min_value = loss_bce_[int(self.thresh * loss_bce.numel())]
        loss_bce = loss_bce[loss_bce >= min_value]
        loss_bce = loss_bce.mean()
        loss = loss_bce + loss_dice
        return loss

def edgeBCE_Dice_loss2(pred, target, edge,thresh=0.75):
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn = DiceLoss(mode='binary',
                           from_logits=False)
    # 交叉熵
    BinaryCrossEntropy_fn = nn.BCELoss(reduction='none')
    '''
    大量的简单背景样本可能会淹没整个交叉熵损失，
    我们利用在线难例挖掘OHEM过滤掉交叉熵小于设定阈值的样本点。
    再者，为进一步缓解正负样本数量不均衡现象，加入Dice损失，得到我们模型的最终损失函数
    '''
    #
    edge_weight = 4.
    loss_bce = BinaryCrossEntropy_fn(pred, target)
    loss_dice = DiceLoss_fn(pred, target)
    edge[edge == 0] = 1.
    edge[edge == 255] = edge_weight
    loss_bce *= edge
    # OHEM
    loss_bce_, ind = loss_bce.contiguous().view(-1).sort()
    min_value = loss_bce_[int(thresh * loss_bce.numel())]
    loss_bce = loss_bce[loss_bce >= min_value]
    loss_bce = loss_bce.mean()
    loss = loss_bce + loss_dice
    return loss

# if __name__ == '__main__':
#     target=torch.ones((2,1,256,256),dtype=torch.float32)
#     input=(torch.ones((2,1,256,256))*0.9)
#     input[0,0,0,0] = 0.99
#     loss=edgeBCE_Dice_loss(input,target,target*255)
#     print(loss)

