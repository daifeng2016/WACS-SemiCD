import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
import torch.nn.functional as F
from .ABLoss import *
##from losses.SSIM import SSIM
'''
pytorch seg loss
https://blog.csdn.net/CaiDaoqing/article/details/90457197
'''
def N8ASCLoss(probs, size=1):
    _, _, h, w = probs.size()
    softmax = F.softmax(probs, dim=1)
    p = size
    softmax_pad = F.pad(softmax, [p]*4, mode='replicate')#must not be probs
    affinity_group = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:#computing the 8 neighbours except (1,1)
                continue
            affinity_paired = torch.sum(
                softmax_pad[:, :, st_y:st_y+h, st_x:st_x+w] * softmax, dim=1)#as softmax is [0,1], it is no use to further norm, must not be probs [4,512,512] torch.norm(softmax)  torch.norm(softmax,dim=1)
            affinity_group.append(affinity_paired.unsqueeze(1))
    affinity = torch.cat(affinity_group, dim=1)#[4,8,512,512]
    loss = 1.0 - affinity
    return loss.mean()

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCELoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


#===========dice_loss+bce_loss==================
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3#[4,321,321]
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()  # predict=[4,21,321,321]
        # ==================================for weighted loss 直接使用此类weight效果很差======================================
        #log_p = predict.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)  # [1,1,384,209]==>[1,80256]
        # target_t = target.view(1, -1)
        # target_trans = target_t.clone()
        # pos_index = (target_t > 0)  # [1,80256]  dtype=torch.uint8 >0处为1，其他位置为0
        # neg_index = (target_t == 0)  # [1,80256]
        # target_trans[pos_index] = 1
        # target_trans[neg_index] = 0
        # pos_index = pos_index.data.cpu().numpy().astype(bool)
        # neg_index = neg_index.data.cpu().numpy().astype(bool)  # 转换为bool后统计正负样本值
        # weight = torch.Tensor(c).fill_(0)  # [1,80256]
        # weight = weight.numpy()
        # pos_num = pos_index.sum()  # 13061
        # neg_num = neg_index.sum()  # 67195
        # sum_num = pos_num + neg_num
        # weight[0] = 1.0-neg_num * 1.0 / sum_num
        # weight[1] = 1.0-pos_num * 1.0 / sum_num
        # weight = torch.from_numpy(weight)
        # weight = weight.cuda()
        # ==============================================================
        target_mask = (target >= 0) * (target != self.ignore_label)#[4,321,321] target>=0 ==>satisfy==1 or 0
        target = target[target_mask]#[412164]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()#[4,21,321,321]==>[4,321,21,321]==>[4,321,321,21] [8655444]
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)#target_mask.view(n, h, w, 1): [4,321,321]==>[4,321,321,1] ==>[4,321,321,21]==>[8655444]==>[412164,21]
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        #self.bce_loss =CrossEntropy2d()
        self.weight=2.0

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)#对二维或多维矩阵的所有元素求和
            j = self.weight*torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)#only for batch=1
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b
#====================for genearized dice loss========================
from models.utils import one_hot
from torch.autograd import Variable
class gen_dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(gen_dice_loss, self).__init__()
        self.batch = batch
        #self.bce_loss = nn.BCELoss()
        self.bce_loss =CrossEntropy2d()
        self.weight=2.0

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        # if self.batch:
        #     i = torch.sum(y_true)#对二维或多维矩阵的所有元素求和
        #     j = self.weight*torch.sum(y_pred)
        #     intersection = torch.sum(y_true * y_pred)
        # else:
        #     i = y_true.sum(1).sum(1).sum(1)#only for batch=1
        #     j = y_pred.sum(1).sum(1).sum(1)
        #     intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score = (2. * intersection + smooth) / (i + j + smooth)
        #target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        if y_true.dim()!=4:
            y_pred = F.softmax(y_pred, dim=1)
            y_true = Variable(one_hot(y_true.data.cpu())).cuda()
        y_true=y_true.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
        y_pred = y_pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
        # y_true = y_true.view(-1, 2)  # [1,3,4,2]==>[12,2]
        # y_pred = y_pred.view(-1, 2)
        sum_p = y_pred.sum(0)  # [1,2]
        sum_r = y_true.sum(0)  # [1,2]
        sum_pr = (y_pred * y_true).sum(0)  # [1,2]
        weights=1.0-sum_r/sum_r.sum(0)
        #weights = torch.pow(sum_r ** 2 + 1e-6, -1)  # seem not useful, sum_r**2==pow(sum_r,2)
        # weights=1/(sum_r**2+1e-6)
        gene_dice = (2 * (weights * sum_pr).sum(0).sum(0)+smooth) / ((weights * (sum_r + sum_p)).sum(0).sum(0)+smooth)

        return gene_dice.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a+0.05*b
#====================for weighted binary cross_entropy loss================
class weighted_ce_loss(nn.Module):
    def __init__(self, batch=True):
        super(weighted_ce_loss, self).__init__()


    def my_ce_loss(self, y_pred, y_true):
        smooth = 1.0  # may change

        y_true_t = Variable(one_hot(y_true.data.cpu())).cuda()
        #y_true=y_true.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
        y_true_t = y_true_t.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)

        sum_r = y_true_t.sum(0)  # [1,2]

        weights = 1.0 - sum_r / sum_r.sum(0)
        weight1=torch.from_numpy(weights.data.cpu().numpy()).cuda(0)


        return F.cross_entropy(y_pred,y_true,weight1)



    def __call__(self, y_pred, y_true):

        return self.my_ce_loss(y_pred,y_true)

#=============for balanced_sigmoid_cross_entropy==============
class Balanced_CE(nn.Module):
    '''
     balanced_sigmoid_cross_entropy
     input is logits before sigmoid
    '''

    # binary cross entropy loss in 2D
    def __init__(self):
        super(Balanced_CE, self).__init__()
        self.bce_loss=nn.BCEWithLogitsLoss()
    def _get_balanced_sigmoid_cross_entropy(self, x):
        count_neg = torch.sum(1. - x)
        count_pos = torch.sum(x)
        beta = count_neg / (count_neg + count_pos)
        pos_weight = beta / (1 - beta)
        cost = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True,
                                          pos_weight=pos_weight)  # using pos_weight not weight
        return cost, 1 - beta

    def forward(self, input, target):
        loss=0
        if target.sum()>0.0:
           crition_seg,beta_seg=self._get_balanced_sigmoid_cross_entropy(target)
           #loss+=crition_seg(input,target)*beta_seg
           loss += crition_seg(input, target)
        else:
            loss+=self.bce_loss(input,target)
        return loss


class Weighted_MSE(nn.Module):
    '''
     weighted mse loss  as described in  Holistically-Nested Edge Detection
    '''
    # binary cross entropy loss in 2D
    def __init__(self):
        super(Weighted_MSE, self).__init__()
        self.MSE=nn.MSELoss()
    def forward(self,x,out_ae,out_seg):
        '''

        :param x: unl_img
        :param out_ae:
        :param out_seg:
        :return:
        '''
        n, c, h, w = x.size()
        bf_thresh=0.5
        # out_bf=out_seg.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # pos_index=(out_seg>bf_thresh)# obtain true or false array
        # neg_index=(out_seg<=bf_thresh)
        pos_idx=np.zeros(shape=(n,1,h,w),dtype=np.uint8)
        neg_idx=np.zeros(shape=(n,1,h,w),dtype=np.uint8)
        pos_idx[out_seg.data.cpu().numpy()>bf_thresh]=1
        neg_idx[out_seg.data.cpu().numpy() <= bf_thresh] = 1
        pos_num=pos_idx.sum()
        neg_num=neg_idx.sum()
        pos_ratio=pos_num*1./(pos_num+neg_num)
        neg_ratio=1-pos_ratio

        pos_idx=torch.from_numpy(pos_idx).cuda(0)
        neg_idx = torch.from_numpy(neg_idx).cuda(0)

        x_pos=x.masked_select((pos_idx))
        x_neg=x.masked_select((neg_idx))

        out_ae_pos=out_ae.masked_select((pos_idx))
        out_ae_neg = out_ae.masked_select((neg_idx))

        # out_seg_pos = out_seg.masked_selected(torch.from_numpy(pos_idx)).cuda(0)
        # out_seg_neg = out_seg.masked_selected(torch.from_numpy(neg_idx)).cuda(0)

        loss=pos_ratio*self.MSE(out_ae_neg,x_neg)+neg_ratio*self.MSE(out_ae_pos,x_pos)

        return loss




class BCE2D(nn.Module):
    '''
     weighted binary entropy loss  as described in  Holistically-Nested Edge Detection
    '''
    # binary cross entropy loss in 2D
    def __init__(self):
        super(BCE2D, self).__init__()

    def forward(self,input, target):
        n, c, h, w = input.size()
        # assert(max(target) == 1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)  # [1,1,384,209]==>[1,80256]
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t > 0)  # [1,80256]  dtype=torch.uint8 >0处为1，其他位置为0
        neg_index = (target_t == 0)  # [1,80256]
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)  # 转换为bool后统计正负样本值
        weight = torch.Tensor(log_p.size()).fill_(0)  # [1,80256]
        weight = weight.numpy()
        pos_num = pos_index.sum()  # 13061
        neg_num = neg_index.sum()  # 67195
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight = torch.from_numpy(weight)
        weight = weight.cuda()

        loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)

        return loss
#====================================weighted bce+laplace edge loss======================
def _cross_entropy(logits,labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

def _weighted_cross_entropy(logits,labels,alpha=0.5):
    count_neg = torch.sum(1. - labels)
    count_pos = torch.sum(labels)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)
    #alpha=pos_weight

    return torch.mean((1 - alpha) * ((1 - labels) * logits + torch.log(1 + torch.exp(-logits))) + (2 * alpha - 1) * labels * torch.log(1 + torch.exp(-logits)))

class EdgeHoldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        #filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace,requires_grad=False)# requires_grad=False（default=True）导致该卷积核不可训练，直接定义提取边缘特征了   含义是将一个固定不可训练的tensor转换成可以训练的类型parameter
    def torchLaplace(self,x):
        edge = F.conv2d(x,self.laplace.cuda(0),padding=1)#out = F.conv2d(x, w, b, stride=1, padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge
    def forward(self,y_pred,y_true,mode=None):
        #y_pred = nn.Sigmoid()(y_pred)
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = _cross_entropy(y_pred_edge,y_true_edge)

        #seg_loss = _weighted_cross_entropy(y_pred,y_true)

        return edge_loss




#===================================bce+iou+ssim=========================================
def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)#[8,1,224,224]

        # one-hot vector of ground truth
        if c==1:
            one_hot_gt =gt
        else:
            one_hot_gt = one_hot(gt, c)#[8,224,224]==>[8,1,224,224]

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)#[8,1,224,224]
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)#[8,1,224,224]

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)#[8,1,224,224]

        # reshape
        gt_b = gt_b.view(n, c, -1)#[8,1,50176]
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)#[8,1]
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)#[8,1]

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)#不指定任何参数就是所有元素的算术平均值

        return loss
class bce_edge_loss(nn.Module):
    def __init__(self, batch=True,use_edge=False,edge_type='abl',use_wiou=False):
        super(bce_edge_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.use_edge=use_edge
        self.use_wiou=use_wiou
        self.weight=2.0
        self.edge_type=edge_type
        if edge_type=='abl':
            self.edge_loss=ABL()
        else:
            self.edge_loss=EdgeHoldLoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)#对二维或多维矩阵的所有元素求和
            j = self.weight*torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)#only for batch=1
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def _iou(self,pred, target, size_average=True):

        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b

    def weighted_iou(self,pred, target, size_average=True):
        '''
        If the number of object pixels in a batch is low,
        a misclassification of the objects by a few pixels causes a large IoU loss. Thus,
        the conventional IoU loss is multiplied by the ratio of the union area
        ref:Domain Adaptive Transfer Attack-Based Segmentation Networks for Building Extraction From Aerial Images
        :param pred:
        :param target:
        :param size_average:
        :return:
        '''
        b = pred.shape[0]
        pix_Num=pred.shape[1]*pred.shape[2]*pred.shape[3]

        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            #IoU1 = Iand1 / Ior1
            IoU+=(Ior1-Iand1)/pix_Num

            # IoU loss is (1-IoU1)
            #IoU = IoU + (1 - IoU1)

        return IoU / b


    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss


    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)#0.7775
        #b = self._iou(y_pred, y_true)#0.9289
        b=self.weighted_iou(y_pred,y_true)

        #d=self.boundary_loss(y_pred,y_true)#1
        #
        if self.use_edge:
            d = self.edge_loss(y_pred, y_true)  # 0.8*(a+b+c)+0.2*d
            if d is not None:
               return a + b + d
        elif self.use_wiou:
            return b
        return a+b



#====================weighted bce_loss====================================
class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        '''
        weighted_bce_loss(D_out, 
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        '''

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)#[4,1]  # equals to F.relu
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()#[4,1]

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)

#===================for mean-teacher loss=====================
def sigmoid_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    # inputs = F.softmax(inputs, dim=1)
    # if use_softmax:
    #     targets = F.softmax(targets, dim=1)

    # if threshold:
    #     loss_mat = F.mse_loss(inputs, targets, reduction='none')
    #     mask = (targets.max(1)[0] > threshold)
    #     loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
    #     if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
    #     return loss_mat.mean()
    # else:
    #     return F.mse_loss(inputs, targets, reduction='mean')
    #method2
    # num_classes = inputs.size()[1]
    # return F.mse_loss(inputs, targets, size_average=False) / num_classes
    #method3===========
    mse_loss = (inputs-targets)**2
    return mse_loss


def sigmoid_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()

    if threshold:
        loss_mat = F.kl_div(input.log(), targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input.log(), targets, reduction='mean')



def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')

def softmax_mse_loss2(inputs, targets, conf_mask=None, prob_mask=None,use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    loss_mat = F.mse_loss(inputs, targets, reduction='none')
    if conf_mask:
       loss_mat = loss_mat[conf_mask.unsqueeze(1).expand_as(loss_mat)]
    if prob_mask is not None:
        loss_mat=loss_mat*prob_mask.unsqueeze(1)
    if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
    return loss_mat.mean()





def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')

from . import ramps
class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)#用于返回一个对象属性值
        self.current_rampup = 0

    # def __call__(self, epoch, curr_iter):
    #     cur_total_iter = self.iters_per_epoch * epoch + curr_iter
    #     if cur_total_iter < self.rampup_starts:
    #         return 0
    #     self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
    #     return self.final_w * self.current_rampup
    def __call__(self, cur_total_iter):
        #cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)

        return self.final_w * self.current_rampup

#===========for entropy loss======================
#fork from D:\TEST\DownPrj\Semi-DA\SemiSeg-Contrastive-main\trainSSL.py
def entropy_loss(v, mask):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    loss_image = torch.mul(v, torch.log2(v + 1e-30))
    loss_image = torch.sum(loss_image, dim=1)
    loss_image = mask.float().squeeze(1) * loss_image


    percentage_valid_points = torch.mean(mask.float())

    return -torch.sum(loss_image) / (n * h * w * np.log2(c) * percentage_valid_points)

def get_adaptive_binary_mask(logit): ## VOC COCO  #this is for CPLG module
    conf = torch.softmax(logit, dim=1)#[4,2,256,256]
    # import ipdb
    # ipdb.set_trace()
    max_value, _ = torch.max(conf.reshape(logit.shape[0], logit.shape[1], -1), dim=2)#[4,2,65536]=>[8,2]
    # print("============================",max_value.shape,"====================================")
    # print(max_value)
    new_max = torch.where(max_value > 0.95, max_value * 0.96, max_value)#eq.(21)  [4,2]
    thresh = new_max.unsqueeze(-1).unsqueeze(-1)#[8,2,1,1]
    # binary_mask = (conf > thresh*0+0.92)
    binary_mask = (conf > thresh)#[8,2,256,256]
    result = torch.sum(binary_mask, dim=1)#[4,256,256] assign different weight for different pixel
    return result#, max_value

#==========fork from D:\TEST\DownPrj\Semi-DA\AugSeg-main\AugSeg-main\augseg\utils\loss_helper.py=====
import scipy.ndimage as nd
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  2. calculate unsupervised loss
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_unsupervised_loss_by_threshold(predict, target, logits, thresh=0.95):
    batch_size, num_class, h, w = predict.shape
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    return loss.mean(), thresh_mask.float().mean()


def compute_unsupervised_loss_by_threshold_hardness(predict, target, logits, thresh=0.95, hardness_tensor=None):
    batch_size, num_class, h, w = predict.shape
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    loss = F.cross_entropy(predict, target, ignore_index=255, reduction="none")
    if hardness_tensor is None:
        return loss.mean()
    loss = loss.mean(dim=[1,2])
    assert loss.shape == hardness_tensor.shape, "wrong hardness calculation!"
    loss *= hardness_tensor
    return loss.mean()


