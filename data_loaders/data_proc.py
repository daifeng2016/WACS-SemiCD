import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
# import torch modules
import scipy.io as ios
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import mkdir_if_not_exist
from data_loaders.RSCD_dl import RSCD_DL
import random
import cv2
# from .FFTAug.style_transfer import style_transfer
# from .FFTAug.data_agu import data_agu
#from losses.edge_utils import mask_to_onehot, onehot_to_binary_edges
#from .FFTAug.copy_paste import copy_paste_self, copy_paste
#from models.Satt_CD.GAN_model import Generator
#from data_loaders.data_proc import RandomFlip,RandomRotate,RandomShiftScaleRotate,RandomHueSaturationValue,ToTensor,Normalize
#==================================for random augmentation of img an label===================================
#=====================online augmentation: 1) increase augmentation types,  2)save moemory, 3) accelerate training============================
#======================however, the initial sample number should be enough==============================================
mean=[0.4406, 0.4487, 0.4149]
std=[0.1993, 0.1872, 0.1959]
import albumentations as A  # using open-source library for img aug



class RandomCrop(object):
    def __init__(self, phw,scale=1,use_dangle=False):
        self.ph = phw
        self.pw = phw
        self.scale=scale#denote to the upsampling scale
        self.use_dangle=use_dangle
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        ix = random.randint(0, w - self.pw)#not using w - self.pw+1
        iy = random.randint(0, h - self.ph)
        tx,ty=int(ix*self.scale),int(iy*self.scale)
        img=img[iy:iy+self.ph, ix:ix+self.pw,:]
        label=label[ty:ty+th, tx:tx+tw]

        if self.use_dangle:
            dist,angle=sample['dist'],sample['angle']
            dist = dist[ty:ty + th, tx:tx + tw]
            angle = angle[ty:ty + th, tx:tx + tw]
            return {'img':img,'label':label,'name':sample['name'],'dist':dist,'angle':angle}

        return {'img':img,'label':label,'name':sample['name']}

from models.utils import one_hot_raw
class RandomCropWeight(object):
    def __init__(self, ph, pw):
        self.ph = ph
        self.pw = pw
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        #th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        bst_x0 = random.randint(0, w - self.pw)#not using w - self.pw+1
        bst_y0 = random.randint(0, h - self.ph)
        one_hot_label=one_hot_raw(label,num_classes=32)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        weight=[1,
                5,5,5,8,5,
                5,5,5,8,5,
                8,8,8,8,8,
                5,5,5,8,5,
                8,8,8,8,8,8,
                5,5,5,5,8
                ]
        for i in range(try_cnt):
            x0 = random.randint(0, w - self.pw)
            y0 = random.randint(0, h - self.ph)
            _sc=0

            for k in range(32):
                _sc+=weight[k]*one_hot_label[y0:y0+self.ph,x0:x0+self.pw,k].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0

        img=img[y0:y0+self.ph, x0:x0+self.pw,:]
        label=label[y0:y0+self.ph,x0:x0+self.pw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomCropWeight7(object):
    def __init__(self, ph, pw):
        self.ph = ph
        self.pw = pw
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        #th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        bst_x0 = random.randint(0, w - self.pw)#not using w - self.pw+1
        bst_y0 = random.randint(0, h - self.ph)
        one_hot_label1=one_hot_raw(label[:,:,0],num_classes=7)
        one_hot_label2 = one_hot_raw(label[:,:,1], num_classes=7)
        bst_sc = -1
        #try_cnt = random.randint(1, 14)
        try_cnt = 14
        # weight=[1,
        #         5,5,5,8,5,
        #         5,5,5,8,5,
        #         8,8,8,8,8,
        #         5,5,5,8,5,
        #         8,8,8,8,8,8,
        #         5,5,5,5,8
        #         ]
        # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        # weight1=[1,5,5,5,5,5,10]
        # weight2=[1,5,5,5,5,5,10]

        weight1=[0.01,3.0,1.0,1.0,1.0,1.0,9.0]
        weight2 = [0.01, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

        for i in range(try_cnt):
            x0 = random.randint(0, w - self.pw)
            y0 = random.randint(0, h - self.ph)
            _sc=0

            for k in range(7):
                _sc+=weight1[k]*one_hot_label1[y0:y0+self.ph,x0:x0+self.pw,k].sum()
                _sc+=weight2[k]*one_hot_label2[y0:y0+self.ph,x0:x0+self.pw,k].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0

        img=img[y0:y0+self.ph, x0:x0+self.pw,:]
        label=label[y0:y0+self.ph,x0:x0+self.pw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomCropResizeWeight7(object):
    def __init__(self, min_crop_size=100, max_crop_size=480):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        #self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):

        img, label = sample['img'], sample['label']

        out_crop = random.random() < 0.5
        #out_crop=True
        if out_crop:
            h, w = img.shape[:2]  # numpy array
            crop_size = random.randint(self.min_crop_size, self.max_crop_size)
            #crop_size = random.randint(int(h / 1.15), int(h / 0.85))

            bst_x0 = random.randint(0, w - crop_size)  # not using w - self.pw+1
            bst_y0 = random.randint(0, h - crop_size)
            one_hot_label1 = one_hot_raw(label[:, :, 0], num_classes=7)
            one_hot_label2 = one_hot_raw(label[:, :, 1], num_classes=7)
            bst_sc = -1
            # try_cnt = random.randint(1, 14)
            try_cnt = 14
            # weight=[1,
            #         5,5,5,8,5,
            #         5,5,5,8,5,
            #         8,8,8,8,8,
            #         5,5,5,8,5,
            #         8,8,8,8,8,8,
            #         5,5,5,5,8
            #         ]
            # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
            # weight1=[1,5,5,5,5,5,10]
            # weight2=[1,5,5,5,5,5,10]

            weight1 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]
            weight2 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

            for i in range(try_cnt):
                x0 = random.randint(0, w - crop_size)
                y0 = random.randint(0, h - crop_size)
                _sc = 0

                for k in range(7):
                    _sc += weight1[k] * one_hot_label1[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                    _sc += weight2[k] * one_hot_label2[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            img_aug = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label_aug = label[y0:y0 + crop_size, x0:x0 + crop_size,:]

            img_aug = cv2.resize(img_aug, (h, w), interpolation=cv2.INTER_CUBIC)
            label_aug = cv2.resize(label_aug, (h, w), interpolation=cv2.INTER_LINEAR)# cannot set cv2.INTER_CUBIC
            return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]

class RandomCutMix(object):
    def __init__(self, min_crop_size=64, max_crop_size=480,num_class=7):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        #self.scale=scale#denote to the upsampling scale
        self.num_class=num_class
    def __call__(self, sample):

        img, label = sample['img'], sample['label']

        out_crop = random.random() < 0.5
        #out_crop=True
        if out_crop:
            h, w = img.shape[:2]  # numpy array
            crop_size = random.randint(self.min_crop_size, self.max_crop_size)
            #crop_size = random.randint(int(h / 1.15), int(h / 0.85))

            bst_x0 = random.randint(0, w - crop_size)  # not using w - self.pw+1
            bst_y0 = random.randint(0, h - crop_size)
            one_hot_label1 = one_hot_raw(label[:, :, 0], num_classes=self.num_class)
            one_hot_label2 = one_hot_raw(label[:, :, 1], num_classes=self.num_class)
            bst_sc = -1
            # try_cnt = random.randint(1, 14)
            try_cnt = 14


            weight1 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]
            weight2 = [0.1, 3.0, 1.0, 1.0, 1.0, 1.0, 9.0]

            for i in range(try_cnt):
                x0 = random.randint(0, w - crop_size)
                y0 = random.randint(0, h - crop_size)
                _sc = 0

                for k in range(self.num_class):
                    _sc += weight1[k] * one_hot_label1[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                    _sc += weight2[k] * one_hot_label2[y0:y0 + crop_size, x0:x0 + crop_size, k].sum()
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            #=====use cutmix for aug==============
            img1,img2=img[:,:,:3],img[:,:,3:]
            label1,label2=label[:,:,:1],label[:,:,1:]
            img1_aug,img2_aug=img1.copy(),img2.copy()
            label1_aug,label2_aug=label1.copy(),label2.copy()
            img1_aug[y0:y0 + crop_size, x0:x0 + crop_size, :]=img2[y0:y0 + crop_size, x0:x0 + crop_size, :]
            img2_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = img1[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label1_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = label2[y0:y0 + crop_size, x0:x0 + crop_size, :]
            label2_aug[y0:y0 + crop_size, x0:x0 + crop_size, :] = label1[y0:y0 + crop_size, x0:x0 + crop_size, :]

            img_aug = np.concatenate([img1_aug,img2_aug],2)
            label_aug = np.concatenate([label1_aug,label2_aug],2)


            return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]


class RandomMixorScale(object):
    def __init__(self, num_class=7):
        self.num_class=num_class
    def __call__(self, sample):
        img,label = sample['img'],sample['label']
        use_mix=random.random() < 0.5
        #use_mix = random.random() < 0.3#use scale more
        if use_mix:
            aug=RandomCutMix()
            aug_res = aug(sample)
            img,label=aug_res['img'],aug_res['label']
        else:
            aug=RandomScale(use_CD=True)
            aug_res = aug(sample)
            img, label = aug_res['img'], aug_res['label']





        return {'img':img,'label':label,'name':sample['name']}




class ReSize(object):
    def __init__(self, ph, pw,scale=1):
        self.ph = ph
        self.pw = pw
        self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_new=cv2.resize(img,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)
        label_new=cv2.resize(label,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)

        return {'img':img_new,'label':label_new,'name':sample['name']}


class RandomFlip(object):
    def __init__(self, use_dangle=False):

        self.use_dangle = use_dangle
    def __call__(self, sample):
        img,label = sample['img'],sample['label']
        hflip=random.random() < 0.5
        vflip=random.random() < 0.5
        #dfilp=random.random() < 0.5
        if self.use_dangle:
            dist=sample['dist']
            angle=sample['angle']
        if vflip:
            img= np.flipud(img).copy()
            label=np.flipud(label).copy()
            if self.use_dangle:
                dist=np.flipud(sample['dist']).copy()
                angle=np.flipud(sample['angle']).copy()
        if hflip:
            img= np.fliplr(img).copy()
            label = np.fliplr(label).copy()
            if self.use_dangle:
                dist=np.fliplr(sample['dist']).copy()
                angle=np.fliplr(sample['angle']).copy()

        if self.use_dangle:
            return {'img': img, 'label': label, 'name': sample['name'],'dist':dist,'angle':angle}

        return {'img':img,'label':label,'name':sample['name']}
class RandomRotate(object):
    def __init__(self, use_dangle=False):

        self.use_dangle = use_dangle
    def __call__(self, sample):
        img, label = sample['img'],sample['label']
        rot90 = random.random() < 0.5
        rot180 = random.random() < 0.5
        rot270 = random.random() < 0.5
        if self.use_dangle:
            dist=sample['dist']
            angle=sample['angle']

        if rot90:
            img = np.rot90(img,1).copy()
            label=np.rot90(label,1).copy()
            if self.use_dangle:
                dist=np.rot90(sample['dist']).copy()
                angle=np.rot90(sample['angle']).copy()
        if rot180:
            img = np.rot90(img,2).copy()
            label = np.rot90(label,2).copy()
            if self.use_dangle:
                dist=np.rot90(sample['dist'],2).copy()
                angle=np.rot90(sample['angle'],2).copy()
        if rot270:
            img=np.rot90(img,3).copy()
            label = np.rot90(label,3).copy()
            if self.use_dangle:
                dist=np.rot90(sample['dist'],3).copy()
                angle=np.rot90(sample['angle'],3).copy()

        if self.use_dangle:
            return {'img': img, 'label': label, 'name': sample['name'],'dist':dist,'angle':angle}

        return {'img':img,'label':label,'name':sample['name']}



class RandomHueSaturationValue1(object):
    def __call__(self, sample, hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15), u=0.5):
        img1, label = sample['img'], sample['label']
        if np.random.random() < u:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(img1)

            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            # h2, s2, v2 = cv2.split(img2)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)

            h1 += hue_shift
            #h2 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s1 = cv2.add(s1, sat_shift)
            #s2 = cv2.add(s2, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v1 = cv2.add(v1, val_shift)
            #v2 = cv2.add(v2, val_shift)

            img1 = cv2.merge((h1, s1, v1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

            # img2 = cv2.merge((h2, s2, v2))
            # img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

        return {'img':img1,'label':label,'name':sample['name']}


class RandomHueSaturationValue(object):
    def __call__(self, sample, hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15), u=0.5):
        img, label = sample['img'], sample['label']
        img1=img[:,:,:3]
        img2=img[:,:,3:]
        if np.random.random() < u:
        #if True:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(img1)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            h2, s2, v2 = cv2.split(img2)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)

            h1 += hue_shift
            h2 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s1 = cv2.add(s1, sat_shift)
            s2 = cv2.add(s2, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v1 = cv2.add(v1, val_shift)
            v2 = cv2.add(v2, val_shift)

            img1 = cv2.merge((h1, s1, v1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

            img2 = cv2.merge((h2, s2, v2))
            img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

            img=np.concatenate([img1,img2],2)

        return {'img':img,'label':label,'name':sample['name']}



class RandomShiftScaleRotate(object):

        def __call__(self, sample, shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
          img1, label = sample['img'], sample['label']
          if np.random.random() < u:
            height, width, channel = img1.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)


            img1= cv2.warpPerspective(img1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            # img2 = cv2.warpPerspective(img2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
            #                            borderValue=(
            #                                0, 0,
            #                                0,))
            label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

          return {'img': img1, 'label': label, 'name': sample['name']}


#============================================================================================================
#=================================for random augmentation of T12=============================================
#============================================================================================================
class RandomFlipT12(object):
    def __call__(self, sample):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        hflip=random.random() < 0.5
        vflip=random.random() < 0.5
        dfilp=random.random() < 0.5
        # if random.random() > 0.5:
        #     img_LR = np.fliplr(img_LR).copy()
        #     img_HR = np.fliplr(img_HR).copy()
        if vflip:
            img_T1 = np.flipud(img_T1).copy()
            img_T2 = np.flipud(img_T2).copy()
            label=np.flipud(label).copy()
        if hflip:
            img_T1 = np.fliplr(img_T1).copy()
            img_T2 = np.fliplr(img_T2).copy()
            label = np.fliplr(label).copy()
        if dfilp:
            img_T1=cv2.flip(img_T1,-1)
            img_T2 = cv2.flip(img_T2, -1)
            label = cv2.flip(label, -1)

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomRotateT12(object):
    def __call__(self, sample):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        rot90 = random.random() < 0.5
        rot180 = random.random() < 0.5
        rot270 = random.random() < 0.5

        if rot90:
            img_T1 = np.rot90(img_T1,1).copy()
            img_T2 = np.rot90(img_T2,1).copy()
            label=np.rot90(label,1).copy()
        if rot180:
            img_T1 = np.rot90(img_T1,2).copy()
            img_T2 = np.rot90(img_T2,2).copy()
            label = np.rot90(label,2).copy()
        if rot270:
            img_T1=np.rot90(img_T1,3).copy()
            img_T2 = np.rot90(img_T2,3).copy()
            label = np.rot90(label,3).copy()

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomShiftScaleRotateT12(object):
    def __call__(self, sample,shift_limit=(-0.0, 0.0),
                               scale_limit=(-0.0, 0.0),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.0, 0.0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        img_T1, img_T2,label = sample['imgT1'], sample['imgT2'],sample['label']
        if random.random()<u:
            height, width, channel = img_T1.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            img1 = cv2.warpPerspective(img_T1, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))
            img2 = cv2.warpPerspective(img_T2, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))
            mask = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

        return {'imgT1':img_T1,'imgT2':img_T2,'label':label}

class RandomHueSaturationValueT12(object):
    def __call__(self, sample,hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'],sample['label']
        if random.random()<u:
            img1 = cv2.cvtColor(img_T1, cv2.COLOR_BGR2HSV)
            h1, s1, v1 = cv2.split(img1)

            img2 = cv2.cvtColor(img_T2, cv2.COLOR_BGR2HSV)
            h2, s2, v2 = cv2.split(img2)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)

            h1 += hue_shift
            h2 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s1 = cv2.add(s1, sat_shift)
            s2 = cv2.add(s2, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v1 = cv2.add(v1, val_shift)
            v2 = cv2.add(v2, val_shift)

            img1 = cv2.merge((h1, s1, v1))
            img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2BGR)

            img2 = cv2.merge((h2, s2, v2))
            img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

            img_T1=img1.copy()
            img_T2=img2.copy()

        return  {'imgT1':img_T1,'imgT2':img_T2,'label':label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'], sample['label']
        label=np.expand_dims(label,axis=-1)
        img_T1_tensor=torch.from_numpy(img_T1.transpose((2,0,1)))
        img_T2_tensor= torch.from_numpy(img_T2.transpose((2, 0, 1)))
        label_tensor = torch.from_numpy(label.transpose((2, 0, 1)))
        img_T1_tensor= img_T1_tensor.float().div(255)
        img_T2_tensor= img_T2_tensor.float().div(255)
        label_tensor=label_tensor.float().div(255)
        return {'imgT1':img_T1_tensor,'imgT2':img_T2_tensor,'label':label_tensor}
class ToTensor_Test(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img_T12= sample['image']

        img_T12_tensor=torch.from_numpy(img_T12.transpose((2,0,1)))

        img_T12_tensor= img_T12_tensor.float().div(255)
        return {'image':img_T12_tensor,'name':sample['name']}



class ToTensor_Sense(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self,use_label255=False,use_rgb=False,use_label32=False,use_dangle=False,use_edge=False):


        # self.use_label_rgb=use_label_rgb
        self.use_label32=use_label32
        self.use_label255=use_label255
        self.use_rgb=use_rgb
        self.use_dangle=use_dangle
        self.use_edge=use_edge
    def __call__(self, sample):
        img= sample['img']
        timg=torch.from_numpy(img.transpose((2,0,1)))#[512,512,6]==>[6,512,512]
        timg= timg.float().div(255)

        label = sample['label']#[512,512,2]
        if isinstance(label,np.ndarray):
            if self.use_label255:
                tlabel = torch.from_numpy(label.transpose((2, 0, 1)).copy())
                tlabel = tlabel.float().div(255)
            elif self.use_rgb:
                tlabel = torch.from_numpy(label.transpose((2, 0, 1)))
            else:
                tlabel = torch.from_numpy(label)

        else:
            #tlabel=None# cause exception using none
            tlabel = torch.from_numpy(label)

        if self.use_dangle:
            pass
            tdist=torch.from_numpy(sample['dist'])
            tangle=torch.from_numpy(sample['angle'])
            return {'img':timg,'label':tlabel,'name':sample['name'],'dist':tdist,'angle':tangle}
        if self.use_edge:
            return {'img': timg, 'label': tlabel, 'edge':sample['edge'],'name': sample['name']}
        return {'img':timg,'label':tlabel,'name':sample['name']}



class Normalize_T12(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self, mean, std):
        # self.mean = torch.from_numpy(np.array(mean))
        # self.std =torch.from_numpy(np.array(std))
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        img1= sample['img'][0:3]
        img2=sample['img'][3:6]
        # img-=self.mean
        # img/=self.std
        self.mean = torch.Tensor(self.mean).view(3, 1, 1)#must have .view(3, 1, 1), broadcat can work only the two tensors have same ndims
        self.std=torch.Tensor(self.std).view(3, 1, 1)
        img1=img1.sub_(self.mean).div_(self.std)
        img2= img2.sub_(self.mean).div_(self.std)
        img12=torch.cat([img1,img2],dim=0)

        return {'img':img12,'label':sample['label'],'name':sample['name']}

class Normalize_BR(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self, mean, std):
        # self.mean = torch.from_numpy(np.array(mean))
        # self.std =torch.from_numpy(np.array(std))
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        img= sample['img']
        # img-=self.mean
        # img/=self.std
        self.mean = torch.Tensor(self.mean).view(3, 1, 1)#must have .view(3, 1, 1), broadcat can work only the two tensors have same ndims
        self.std=torch.Tensor(self.std).view(3, 1, 1)
        img=img.sub_(self.mean).div_(self.std)


        return {'img':img,'label':sample['label'],'name':sample['name']}



class Normalize(object):
    def __call__(self, sample,mean,std):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'], sample['label']
        img_T1-=mean[0:3]
        img_T1/=std[0:3]
        img_T2-=mean[3:6]
        img_T2/=std[3:6]
        return {'imgT1': img_T1, 'imgT2': img_T2, 'label': label}
#####################################################################
#========================for data loader============================#
#####################################################################
transform = transforms.Compose([
    # RandomShiftScaleRotate(),
    # RandomFlip(),
    # RandomRotate(),
    # ToTensor()#note that new dict is used for input, hence transforms.ToTensor()  should not be used

    transforms.ToTensor()
    #transforms.Normalize(mean=[.5,.5,.5,.5,.5,.5],std=[.5,.5,.5,.5,.5,.5])#[-1,1]
])


target_transform = transforms.Compose([
    transforms.ToTensor()  ##将图片转换为Tensor,归一化至[0,1]
])







#==============================================================================
class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=2):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array((label.byte().squeeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        return torch.from_numpy(ohlabel)
#=====================random augmentation for img12 using import albumentations as A=============================
import albumentations as A
class RandomShiftScale(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:
            # aug = A.ShiftScaleRotate(p=1)#p must be 1 , or else aug_op for img1 and img2 may be different
            # img1_aug = aug(image=img[:, :, :3], mask=label)
            # img2_aug = aug(image=img[:, :, 3:], mask=label)
            # img_aug = np.concatenate([img1_aug['image'], img2_aug['image']], 2)
            # label_aug = img1_aug['mask']
            aug=A.Compose([
                 A.ShiftScaleRotate(0.5)],additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)
            # img1_aug = aug_op['image']
            # img2_aug = aug_op['image2']
            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']

        else:
            aug = A.ShiftScaleRotate(p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug=aug_f['image']
            label_aug=aug_f['mask']

        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}

class RandomScale(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:

            # aug=A.Compose([
            #      A.RandomScale(p=0.5)],additional_targets={'image2': 'image'})#will change input size

            aug = A.Compose([
                A.RandomSizedCrop(min_max_height=(100, 500), height=512, width=512, interpolation=2, p=0.5)], additional_targets={'image2': 'image'})

            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)

            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']

        else:
            aug = A.RandomSizedCrop(min_max_height=(50, 500), height=512, width=512, interpolation=2, p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug=aug_f['image']
            label_aug=aug_f['mask']

        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}
class RandomTranspose(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']


        if self.use_CD:

            aug = A.Compose([
                A.Transpose(p=0.5)], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3], image2=img[:, :, 3:], mask=label)
            img_aug = np.concatenate([aug_op['image'], aug_op['image2']], 2)
            label_aug = aug_op['mask']
        else:
            aug = A.Transpose(p=0.5)
            aug_f = aug(image=img, mask=label)
            img_aug = aug_f['image']
            label_aug = aug_f['mask']


        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}
class RandomNoise(object):#can not use oneof for imgs12, which may lead to differnt aug_op for img1 and img2
    def __init__(self, use_CD=False):
        self.use_CD = use_CD
    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'],img[:,:,3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.5)
            aug_f = aug(image=img)
            img_aug=aug_f['image']


        return {'img': img_aug, 'label': label, 'name': sample['name']}
class RandomColor(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD
    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'], img[:, :, 3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss()
            ], p=0.5)
            aug_f=aug(image=img)
            img_aug=aug_f['image']


        return {'img': img_aug, 'label': label, 'name': sample['name']}


class RandomColor2(object):
    def __init__(self, use_CD=False):
        self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']



        if self.use_CD:
            aug = A.Compose([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift()], additional_targets={'image2': 'image'})
            aug_op = aug(image=img[:, :, :3])
            img_aug = np.concatenate([aug_op['image'], img[:, :, 3:]], 2)
            #label1_aug = label
        else:
            aug = A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift()
            ], p=0.5)
            aug_f = aug(image=img)
            img_aug = aug_f['image']

        return {'img': img_aug, 'label': label, 'name': sample['name']}









class RandomMix(object):
    # def __init__(self, use_CD=False):
    #     self.use_CD = use_CD

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        b_mix = random.random() < 0.5
        f_mix = random.random() < 0.5
        # mix two periods of images using label map
        img1 = img[:, :, :3]
        label1 = label[:, :, 0]#[512,512],label[:, :, :1]==>[512,512,1]
        img2 = img[:, :, 3:]
        label2 = label[:, :, 1]
        if b_mix:
            # change background, label_mix is unchanged
            img1_mix = img1.copy()
            label1_mix = label1.copy()
            img2_mix = img2.copy()
            label2_mix = label2.copy()

            img1_mix[label1 == 0] = img2[label2 == 0]
            img2_mix[label2 == 0] = img1[label2 == 0]

            img= np.concatenate([img1_mix, img2_mix], 2)
            label= np.concatenate([label1_mix[:, :, None], label2_mix[:, :, None]], 2)
            return {'img': img, 'label': label, 'name': sample['name']}
        if f_mix:
            # change foreground, label_mix is changed
            img1_mix = img1.copy()
            label1_mix = label1.copy()
            img2_mix = img2.copy()
            label2_mix = label2.copy()

            img1_mix[label1 > 0] = img2[label2 > 0]
            img2_mix[label2 > 0] = img1[label2 > 0]
            label1_mix[label1 > 0] = label2[label1 > 0]
            label2_mix[label2 > 0] = label1[label2 > 0]

            img= np.concatenate([img1_mix, img2_mix], 2)
            label= np.concatenate([label1_mix[:, :, None], label2_mix[:, :, None]], 2)
            return {'img': img, 'label': label, 'name': sample['name']}


        return {'img': img, 'label': label, 'name': sample['name']}
#=========***********************************************************************************************======
#===============================for semicd dataset 2023-9-26, forked from unimatch=============================
import math
from copy import deepcopy
from .transform import *


def get_patch_tensor(h=256,w=256,mask_gap=16):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask0_tensor = torch.arange(0,h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))
    mask1_tensor=torch.randperm(h_gap_num*w_gap_num).float().reshape((h_gap_num,w_gap_num))
    #mask1_tensor=torch.from_numpy(np.array([3,2,0,1])).float().reshape((h_gap_num, w_gap_num))

    mask0_tensor=F.interpolate(mask0_tensor.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)
    mask1_tensor = F.interpolate(mask1_tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(
        0).squeeze(0)

    return mask0_tensor.numpy(),mask1_tensor.numpy(),h_gap_num*w_gap_num


def get_patch_tensor2(h=256,w=256,mask_gap=32):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask0_tensor = torch.arange(0,h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))

    #mask1_tensor=torch.from_numpy(np.array([3,2,0,1])).float().reshape((h_gap_num, w_gap_num))
    rand_list=torch.randperm(h_gap_num*w_gap_num)
    mask1_tensor = rand_list.float().reshape((h_gap_num, w_gap_num))
    mask0_tensor=F.interpolate(mask0_tensor.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)
    # mask1_tensor = F.interpolate(mask1_tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(
    #     0).squeeze(0)

    return mask0_tensor.numpy(),rand_list.numpy(),h_gap_num*w_gap_num

def get_mask_tensor(h=256,w=256,mask_gap=16,mask_rate=0.75):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask_tensor_small=torch.randperm(h_gap_num*w_gap_num).float().reshape((h_gap_num,w_gap_num))
    divide_threshold=h_gap_num * w_gap_num * mask_rate
    mask_tensor_small[mask_tensor_small<divide_threshold]=0.0
    mask_tensor_small[mask_tensor_small>=divide_threshold]=1.0
    mask_tensor=F.interpolate(mask_tensor_small.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)

    return mask_tensor

def get_patch_tensor(h=256,w=256,mask_gap=16):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask0_tensor = torch.arange(0,h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))
    mask1_tensor=torch.randperm(h_gap_num*w_gap_num).float().reshape((h_gap_num,w_gap_num))
    #mask1_tensor=torch.from_numpy(np.array([3,2,0,1])).float().reshape((h_gap_num, w_gap_num))

    mask0_tensor=F.interpolate(mask0_tensor.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)
    mask1_tensor = F.interpolate(mask1_tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(
        0).squeeze(0)

    return mask0_tensor.numpy(),mask1_tensor.numpy(),h_gap_num*w_gap_num

def get_patch_tensor2(h=256,w=256,mask_gap=16):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask0_tensor = torch.arange(0,h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))

    #mask1_tensor=torch.from_numpy(np.array([3,2,0,1])).float().reshape((h_gap_num, w_gap_num))
    rand_list=torch.randperm(h_gap_num*w_gap_num)
    mask1_tensor = rand_list.float().reshape((h_gap_num, w_gap_num))
    mask0_tensor=F.interpolate(mask0_tensor.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)
    mask1_tensor = F.interpolate(mask1_tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(
        0).squeeze(0)

    return mask0_tensor.numpy(),rand_list.numpy(),h_gap_num*w_gap_num


def get_patch_tensor_list(h=256,w=256,mask_gap=16,num=1):
    if h%mask_gap!=0 or w%mask_gap!=0:
        raise Exception("h and w should be integral multiple mask_gap")
    h_gap_num,w_gap_num=int(h/mask_gap),int(w/mask_gap)
    mask0_tensor = torch.arange(0,h_gap_num * w_gap_num).float().reshape((h_gap_num, w_gap_num))
    #===============for rand_list for a batch of images=============
    rand_list=[]
    for i in range(num):
        rand_list.append(torch.randperm(h_gap_num*w_gap_num))
    #
    mask0_tensor=F.interpolate(mask0_tensor.unsqueeze(0).unsqueeze(0),size=(h,w),mode='nearest').squeeze(0).squeeze(0)


    return mask0_tensor,torch.stack(rand_list),h_gap_num*w_gap_num

class SemiCDDataset(Dataset):
    def __init__(self, root, mode, size=None, id_path=None, nsample=None,dataset='LEVIR'):
        #self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.dataset=dataset


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()  # return the total lines of the text file
            if mode == 'train_l' and nsample is not None:
                self.set_num=len(self.ids)
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
            self.img_dir='train'

        elif mode == 'val':
            val_path=os.path.join(root,'splits','val.txt')
            self.img_dir='val'
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()
        else:
            val_path=os.path.join(root,'splits','test.txt')
            self.img_dir='test'
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        if self.dataset=='LEVCD':
            imgA = Image.open(os.path.join(self.root, self.img_dir, 'T1', id)).convert('RGB')  # [256,256]
            imgB = Image.open(os.path.join(self.root, self.img_dir, 'T2', id)).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root, self.img_dir, 'label', id)))
        else:
            imgA = Image.open(os.path.join(self.root, 'T1', id)).convert('RGB')  # [256,256]
            imgB = Image.open(os.path.join(self.root,  'T2', id)).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root,  'label', id)))


        #===========for style stransfer pdf 2024-2-3==================
        #=========leads to lower performance when use style_transfer
        # imgA=np.array(imgA)
        # imgB=np.array(imgB)
        # imgA2B=style_transfer(imgA,imgB)
        # imgA = Image.fromarray(imgA2B.astype(np.uint8))
        # imgB = Image.fromarray(imgB.astype(np.uint8))
        #=============================================================
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))


        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id
        if self.mode=='test':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            img_name=id.split('.')[0]
            sample = {'img': torch.cat([imgA,imgB],dim=0), 'label':mask, 'name': img_name}
            return sample

        # ==image-level pertubations to serve as weak pertubations=============
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))  # [277,277]
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)  # [256,256]
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)  # [256,256]




        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask

        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)
        # ============futher feature-level pertubations to serve as strong pertubations=================
        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=0.5)  # [256,256]

        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=0.5)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=0.5)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))#[256,256]
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 255] = 255

        return normalize(imgA_w), normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
               normalize(imgA_s2), normalize(imgB_s2), ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)









class CB_SemiCDDataset(Dataset):
    def __init__(self, root, mode, size=None, id_path=None, nsample=None,dataset='LEVIR'):
        #self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.dataset=dataset


        if mode == 'train_l':
            with open(id_path, 'r') as f:
                self.ids= f.read().splitlines()  # return the total lines of the text file
            # if mode == 'train_l' and nsample is not None:
            #     self.set_num=len(self.ids)
            #     self.ids *= math.ceil(nsample / len(self.ids))
            #     self.ids = self.ids[:nsample]
            self.num_l=len(self.ids)
            self.num_u=nsample
            self.generate_changed_ratio()
            self.generate_repeat_ids()
            self.set_num = len(self.ids)
            self.ids_new *= math.ceil(nsample / len(self.ids_new))
            self.ids = self.ids_new[:nsample]
            self.img_dir='train'
        elif mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids= f.read().splitlines()
            self.img_dir = 'train'
        else:
            val_path=os.path.join(root,'splits','val.txt')
            self.img_dir='val'
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()

    def generate_changed_ratio(self):
        self.ratio_list=[]
        for img_file in self.ids:
            if self.dataset=='LEVCD':
                img_path = os.path.join(self.root, 'train/label', img_file)
            else:#for whucd
                img_path=os.path.join(self.root, 'label', img_file)
            label = cv2.imread(img_path, 0)
            count = np.count_nonzero(label)
            height, width = label.shape
            changed_ratio = count * 1.0 / (height * width)
            self.ratio_list.append(changed_ratio)

        #max_ratio = np.max(ratio_list)
        self.ratio_list = np.array(self.ratio_list)
        torch_list = torch.from_numpy(self.ratio_list)
        torch_list_unique = torch.unique(torch_list)
        self.median_ratio = torch.median(torch_list_unique).numpy()


    def generate_repeat_ids(self):
        '''
          repeat_num=max(1,ratio/median_ratio)

         repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))#replicate dataset_idx like [1]*3=[1] [1] [1]
        self.repeat_indices = repeat_indices
        '''
        #repeat_factors=[]
        self.ids_new=[]
        for index,cur_ids in enumerate(self.ids):
            repeat_factor=max(1.0, self.ratio_list[index]/self.median_ratio)
            self.ids_new.extend([cur_ids]*math.ceil(repeat_factor))

        random.shuffle(self.ids_new)


    def __getitem__(self, item):

        # if self.mode=='train_l':
        #     item=int(item%len(self.ids_new))
        #     id=self.ids_new[item]
        # else:
        #     id = self.ids[item]

        id = self.ids[item]

        # imgA = Image.open(os.path.join(self.root, self.img_dir,'T1', id)).convert('RGB')#[256,256]
        # imgB = Image.open(os.path.join(self.root, self.img_dir,'T2', id)).convert('RGB')
        # mask = np.array(Image.open(os.path.join(self.root, self.img_dir,'label', id)))
        if self.dataset=='LEVCD':
            imgA = Image.open(os.path.join(self.root, self.img_dir, 'T1', id)).convert('RGB')  # [256,256]
            imgB = Image.open(os.path.join(self.root, self.img_dir, 'T2', id)).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root, self.img_dir, 'label', id)))
        else:
            imgA = Image.open(os.path.join(self.root, 'T1', id)).convert('RGB')  # [256,256]
            imgB = Image.open(os.path.join(self.root,  'T2', id)).convert('RGB')
            mask = np.array(Image.open(os.path.join(self.root,  'label', id)))



        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id
        # ==image-level pertubations to serve as weak pertubations=============
        imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))  # [277,277]
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size)  # [256,256]
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)  # [256,256]

        if self.mode == 'train_l':
            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask

        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)
        # ============futher feature-level pertubations to serve as strong pertubations=================
        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=0.5)  # [256,256]

        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=0.5)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=0.5)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))#[256,256]
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 255] = 255

        return normalize(imgA_w), normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
               normalize(imgA_s2), normalize(imgB_s2), ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        # if self.mode=='train_l':
        #     return len(self.ids_new)
        # else:
        return len(self.ids)






