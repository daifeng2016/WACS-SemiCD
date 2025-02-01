import torch
import numpy as np
import random
import torch.nn.functional as F
#===================for training utils function==========================
'''
        #use different alpha for teacher update
        m = 1 - (1 - 0.995) * (math.cos(math.pi * i_iter / num_iterations) + 1) / 2
        ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=m, iteration=i_iter)
'''
def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    """

    Args:
        ema_model: model to update
        model: model from which to update parameters
        alpha_teacher: value for weighting the ema_model
        iteration: current iteration

    Returns: ema_model, with parameters updated follwoing the exponential moving average of [model]

    """
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model

#================================================
#fork from D:\TEST\DownPrj\Semi-DA\DSSN-main\voc\util\utils.py
# def get_adaptive_binary_mask(logit): ## the input is logit begore norm
#     conf = torch.softmax(logit, dim=1)
#     # import ipdb
#     # ipdb.set_trace()
#     max_value, _ = torch.max(conf.reshape(logit.shape[0], logit.shape[1], -1), dim=2)
#     # print("============================",max_value.shape,"====================================")
#     # print(max_value)
#     new_max = torch.where(max_value > 0.95, max_value * 0.96, max_value)#eq.(21)
#     thresh = new_max.unsqueeze(-1).unsqueeze(-1)
#     # binary_mask = (conf > thresh*0+0.92)
#     binary_mask = (conf > thresh)
#     result = torch.sum(binary_mask, dim=1)
#     return result#, max_value

def get_adaptive_binary_mask(logit,thresh=0.95): ## VOC COCO  #this is for CPLG module
    conf = torch.softmax(logit, dim=1)#[4,2,256,256]
    # import ipdb
    # ipdb.set_trace()
    max_value, _ = torch.max(conf.reshape(logit.shape[0], logit.shape[1], -1), dim=2)#[4,2,65536]=>[8,2]
    # print("============================",max_value.shape,"====================================")
    # print(max_value)we establish a class-wise threshold ???? by multiplying the maximum probability by ??%. Pixels exceeding this class-wise thresh- old are selected,
    # exclude pixels with a low maximum probability since they indicate lower prediction confidence
    new_max = torch.where(max_value > thresh, max_value * 0.95, max_value)#eq.(21)  [4,2]
    thresh = new_max.unsqueeze(-1).unsqueeze(-1)#[8,2,1,1]
    # binary_mask = (conf > thresh*0+0.92)
    binary_mask = (conf > thresh)#[8,2,256,256]
    result = torch.sum(binary_mask, dim=1)#[4,256,256] assign different weight for different pixel
    return result#, max_value

def get_adaptive_binary_mask0(logit): ## VOC COCO  #this is for CPLG module
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


def get_adaptive_binary_mask2(conf,mask): ## the input is logit begore norm
    #conf = torch.softmax(logit, dim=1)
    # import ipdb
    # ipdb.set_trace()
    #max_value, _ = torch.max(conf.reshape(logit.shape[0], logit.shape[1], -1), dim=2)
    # print("============================",max_value.shape,"====================================")
    # print(max_value)
    new_max = torch.where(mask > 0.95, mask * 0.96, mask)#eq.(21)
    thresh = new_max.unsqueeze(-1).unsqueeze(-1)
    # binary_mask = (conf > thresh*0+0.92)
    binary_mask = (conf > thresh)
    result = torch.sum(binary_mask, dim=1)
    return result#, max_value


#=======================================================
def sigmoid_ramp_up(iter, max_iter):
    """

    Args:
        iter: current iteration
        max_iter: maximum number of iterations to perform the rampup

    Returns:
        returns 1 if iter >= max_iter
        returns [0,1] incrementally from 0 to max_iters if iter < max_iter

    """
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(iter) / float(max_iter)) ** 2)

#=============fork from D:\TEST\DownPrj\Semi-DA\AugSeg-main\AugSeg-main\augseg\dataset\augs_ALIA.py==========
# # # # # # # # # # # # # # # # # # # # #
# # 0 random box
# # # # # # # # # # # # # # # # # # # # #
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # #
# # 1 cutmix label-adaptive
# # # # # # # # # # # # # # # # # # # # #
def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(labeled_mask)

    # 1) get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # 2) get box
    l_bbx1, l_bby1, l_bbx2, l_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(8, 2))
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # 3) labeled adaptive, these less confident
    # unlabeled samples are more likely to be
    # aided (mixed) by these confident labeled samples
    for i in range(0, mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            mix_unlabeled_image[i, :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_image[u_rand_index[i], :, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]

            mix_unlabeled_target[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_mask[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]

            mix_unlabeled_logits[i, l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]] = \
                labeled_logits[u_rand_index[i], l_bbx1[i]:l_bbx2[i], l_bby1[i]:l_bby2[i]]

    # 4) copy and paste
    for i in range(0, unlabeled_image.shape[0]):
        unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_target[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits

    return unlabeled_image, unlabeled_mask, unlabeled_logits

#============generate cutbox using pseudo label=============
from skimage.measure import label, regionprops, find_contours
""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):#[1019,1214]
    bboxes = []

    mask = mask_to_border(mask)#find borders [1019,1214], must have this line, otherwise the regionnum is not correct
    lbl = label(mask)#实现连通区域标记 [1019,1214]
    props = regionprops(lbl)#每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等，则需要调用measure子模块的regionprops（）函数
    #print('region num=%d'% len(props))
    area_thresh=5000
    #area_up_limit=0.7*(mask.shape[0]*mask.shape[1])
    for prop in props:
        if prop.area>area_thresh:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append([x1, y1, x2, y2])
    #print('box num=%d' % len(bboxes))
    return bboxes


def mask_to_bbox2(mask):#[1019,1214]
    bboxes = []

    mask = mask_to_border(mask)#find borders [1019,1214], must have this line, otherwise the regionnum is not correct
    lbl = label(mask)#实现连通区域标记 [1019,1214]
    props = regionprops(lbl)#每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等，则需要调用measure子模块的regionprops（）函数
    #print('region num=%d'% len(props))
    area_thresh=5000
    area_limit=0.65*(mask.shape[0]*mask.shape[1])
    for prop in props:
        if prop.area>area_thresh and prop.area<area_limit:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append([x1, y1, x2, y2])
    #print('box num=%d' % len(bboxes))
    return bboxes



def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_cutmix_box2(img_size, ratio=2):#not work for CD

    mask = torch.zeros(img_size, img_size)
    cut_area = img_size* img_size/ ratio
    w = np.random.randint(img_size / ratio + 1, img_size)
    h = np.round(cut_area / w)

    x_start = np.random.randint(0, img_size - w + 1)
    y_start = np.random.randint(0, img_size - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)
    #mask = torch.ones(img_size,img_size)
    mask[y_start:y_end, x_start:x_end] = 1

    return mask



def generate_pseudo_box(mask):
    '''
    mask:binary img,numpy array:[0,255]
    '''
    batch_size,_,img_size=mask.shape
    """ Detecting bounding boxes """
    #bboxes = mask_to_bbox(mask)
    #box_mask = torch.zeros(img_size, img_size)
    #box_list=[]
    box_mask1_list=[]
    box_mask2_list=[]
    for i in range(batch_size):
        bboxes=mask_to_bbox2(mask[i])
        #box_list.append(box_list)
        # box_mask1 = obtain_cutmix_box(img_size)
        # box_mask2 = obtain_cutmix_box(img_size)

        box_mask1 = obtain_cutmix_box(img_size)
        box_mask2 = obtain_cutmix_box(img_size)
        for (x1, y1, x2, y2) in bboxes:
            box_mask1[y1:y2, x1:x2] = 1
            box_mask2[y1:y2, x1:x2] = 1

        box_mask1_list.append(box_mask1)
        box_mask2_list.append(box_mask2)#8[256,256]
    #===============in case changed areas are too small===============
    box_mask1_tensor=torch.stack(box_mask1_list,dim=0)#[8,256,256]
    box_mask2_tensor = torch.stack(box_mask2_list, dim=0)

    return box_mask1_tensor,box_mask2_tensor


#===============
import torch.nn as nn
class SpatialRefine(nn.Module):#ref Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation
    def __init__(self, in_ch,out_ch,kernel_size=1,padding=0):
        super(SpatialRefine, self).__init__()
        self.conv_fea = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred,fea):
        n, c, h, w = fea.size()#[1,32,128,128]
        n, c1, h1, w1 = pred.size()#[1,3,128,128]
        #cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.conv_fea(fea)  # self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        f = f.view(n, -1, h * w)#[1,32,16384]
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)#[1,32,16384]  torch.norm(f, dim=1, keepdim=True) + 1e-5==>[1,1,26384]   feature normalization  torch.norm(input, p=2) → float

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)#[1,16384,16384]
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)##[1,16384,16384] aff norm  torch.sum(aff, dim=1, keepdim=True) + 1e-5:[1,1,16384]
        pred_rv = torch.matmul(pred.view(n,c1,-1), aff).view(n, -1, h, w)#[1,3,128,128]

        return pred_rv

#====================fork from D:\TEST\DownPrj\Semi-DA\dual-teacher-main\dual-teacher-main\tools\train.py========
def update_ema(model_teacher, model, alpha_teacher, iteration):
    with torch.no_grad():
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
        for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]


#================fork from D:\TEST\DownPrj\Semi-DA\AD-MT-main\AD-MT-main\code\train_utils.py
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  1. alternate updating
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class AlternateUpdate:
    def __init__(self, alternate_period, initial_flag=True, flag_random=False):
        self.alternate_period = alternate_period
        self.flag_alternate = initial_flag
        self._counter = 0
        self.flag_random = flag_random
        if self.flag_random:
            self.random_alternate_period = np.random.randint(1, self.alternate_period + 1)
        else:
            self.random_alternate_period = self.flag_alternate

    def reset(self, alternate_period, initial_flag=True, flag_random=False):
        self._counter = 0
        self.alternate_period = alternate_period
        self.flag_alternate = initial_flag
        self.flag_random = flag_random

        if self.flag_random:
            self.random_alternate_period = np.random.randint(1, self.alternate_period + 1)
        else:
            self.random_alternate_period = self.flag_alternate

    def get_alternate_state(self):
        return self.flag_alternate

    def get_alternate_period(self):
        return self.alternate_period

    def set_alternate_period(self, new_period):
        if new_period > 0:
            self.alternate_period = new_period
            if self.flag_random:
                self.random_alternate_period = np.random.randint(1, self.alternate_period + 1)
            else:
                self.random_alternate_period = self.flag_alternate

    # def set_alternate_period(self, new_period):
    #     if new_period > 0 and new_period != self.get_alternate_period():
    #         self.alternate_period = new_period
    #         self.random_alternate_period = np.random.randint(1, self.alternate_period +1)

    def update(self):
        # assert self._counter < self.alternate_period, f"{self._counter}/{self.alternate_period}"
        self._counter += 1
        if self._counter >= self.random_alternate_period:  # 24
            self.flag_alternate = not self.flag_alternate
            self._counter = 0
            if self.flag_random:
                self.random_alternate_period = np.random.randint(1, self.alternate_period + 1)

#============fork from D:\TEST\DownPrj\Semi-DA\AD-MT-main\AD-MT-main\code\train_utils.py==============
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  2. calculate unsupervised loss from two teachers
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_unsupervised_loss_by_2teachers(
        predict,
        target1, logits1,
        target2, logits2,
        entropy1=None, entropy2=None,
        weight_conflict=1.0,
        mode_conflict="latest",
        flag_t1_update_latest=True,
        thresh=0.95):
    batch_size, num_class, h, w = predict.shape

    # ----------
    # remove ent_fusion, kinda complicate, but similar performance to pixel confidence
    # ----------
    # dealing with conflicts and obtain results
    target, logits, mtx_bool_conflict = get_compromise_pseudo_after_conflict(target1, logits1, target2, logits2,
                                                                             mode_conflict, flag_t1_update_latest,
                                                                             num_class, entropy1, entropy2)

    # final calculations
    thresh_mask = logits.ge(thresh).bool() * (target != 255).bool()
    target[~thresh_mask] = 255
    target_consist = target.clone()
    target_consist[mtx_bool_conflict] = 255
    target_conflct = target.clone()
    target_conflct[~mtx_bool_conflict] = 255

    loss_consist = F.cross_entropy(predict, target_consist, ignore_index=255, reduction="none")
    loss_conflct = F.cross_entropy(predict, target_conflct, ignore_index=255, reduction="none")
    loss = loss_consist + loss_conflct * weight_conflict

    # return conflicts
    # return loss.sum() / thresh_mask.float().sum(), mtx_bool_conflict.float().sum() // batch_size
    return loss.mean(), mtx_bool_conflict.float().sum() // batch_size


def get_compromise_pseudo_after_conflict(target1, logits1, target2, logits2,
                                         mode_conflict="pixel_confidence", entropy1=None, entropy2=None):
    target = target1.clone()
    logits = logits1.clone()
    mtx_bool_conflict = target1 != target2

    if "low_ent"==mode_conflict:
        if entropy1 is not None and entropy2 is not None:
            if "low_ent_all" in mode_conflict:
                tmp_flag = entropy2.sum(dim=[1, 2]) < entropy1.sum(dim=[1, 2])
            else:  # "low_ent_conflict" --> only the conflict region
                tmp_flag = (entropy2 * mtx_bool_conflict.float()).sum(dim=[1, 2]) < (
                            entropy1 * mtx_bool_conflict.float()).sum(dim=[1, 2])
            target[tmp_flag, :, :] = target2[tmp_flag, :, :]
            logits[tmp_flag, :, :] = logits[tmp_flag, :, :]

    # elif "latest"==mode_conflict:
    #     if not flag_t1_update_latest:
    #         target[mtx_bool_conflict] = target2[mtx_bool_conflict]
    #         logits[mtx_bool_conflict] = logits2[mtx_bool_conflict]

    elif "random"==mode_conflict:
        if np.random.random() < 0.5:
            target[mtx_bool_conflict] = target2[mtx_bool_conflict]
            logits[mtx_bool_conflict] = logits2[mtx_bool_conflict]

    elif "pixel_confidence"==mode_conflict:
        if entropy1 is not None and entropy2 is not None:
            bool_better_tea2 = entropy2 < entropy1
        else:
            bool_better_tea2 = logits2 > logits1
        target[bool_better_tea2] = target2[bool_better_tea2]
        logits[bool_better_tea2] = logits2[bool_better_tea2]

    else:
        raise NotImplementedError(
            "conflict mode {} is not supported".format(mode_conflict)
        )

    return target, logits, mtx_bool_conflict


def get_compromise_pseudo_btw_tea_stu(target_tea, logits_tea, target_stu, logits_stu,
                                      mode_conflict='pixel_confidence',
                                      mtx_teacher_conflict=None):
    target = target_tea.clone()  # [12,256,256]
    logits = logits_tea.clone()  # [12,256,256]
    if mtx_teacher_conflict is None:
        mtx_bool_conflict = target_tea != target_stu
    else:
        mtx_bool_conflict_stu = target_tea != target_stu  # [12,256,256]
        mtx_bool_conflict = mtx_bool_conflict_stu & mtx_teacher_conflict  # [12,256,256]

    bool_better_stu = logits_stu > logits_tea  # [12,256,256]
    bool_better_stu_select = bool_better_stu & mtx_bool_conflict  # [12,256,256]
    target[bool_better_stu_select] = target_stu[bool_better_stu_select]  # [12,256,256]
    logits[bool_better_stu_select] = logits_stu[bool_better_stu_select]  # [12,256,256]

    # if "random" in mode_conflict:
    #     if np.random.random() < 0.5:
    #         target[mtx_bool_conflict] = target_stu[mtx_bool_conflict]
    #         logits[mtx_bool_conflict] = logits_stu[mtx_bool_conflict]
    #
    # elif "pixel_confidence" in mode_conflict:



    # elif "always_tea" in mode_conflict:
    #     pass
    #
    # elif "always_stu" in mode_conflict:
    #     target[mtx_bool_conflict] = target_stu[mtx_bool_conflict]
    #     logits[mtx_bool_conflict] = logits_stu[mtx_bool_conflict]
    #
    # else:
    #     raise NotImplementedError(
    #         "conflict mode {} is not supported".format(mode_conflict)
    #     )

    return target, logits, mtx_bool_conflict

#===========for online cutmix with conflict==================
# # # # # # # # # # # # # # # # # # # # #
# # 0. random box
# # # # # # # # # # # # # # # # # # # # #
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()

    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)

    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
        #         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits
