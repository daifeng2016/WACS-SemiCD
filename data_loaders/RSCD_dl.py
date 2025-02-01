# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by D. F. Peng on 2019/6/18
"""

from utils.utils import mkdir_if_not_exist

import numpy as np
import random
import cv2
import math
from tqdm import tqdm
import glob
import os
import json
import logging

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class RSCD_DL(object):
    def __init__(self, config=None):
        #super(RSCD_DL, self).__init__(config) #没有继承类时不要
        #====================load RS data========================
        # self.train_img_path = config.data_dir + '/train/train_img.npy'
        # self.train_label_path = config.data_dir + '/train/train_label.npy'
        # self.test_img_path = config.data_dir + '/test/test_img.npy'
        # self.test_label_path = config.data_dir + '/test/test_label.npy'
        config.img_dir = config.data_dir + '/result/img'
        config.model_dir = config.data_dir + '/result/model'
        config.log_dir = config.data_dir + '/result/log'
        #==============================for train===================================================
        #==============================================================================
        #=================================for Air-CD===================================
        config.train_dir = config.data_dir + '/train'

        #===============for test=======================
        # ====for other_dir test==================
        # config.test_dir=config.data_dir00 + '/test'
        # config.test_pred_dir = config.data_dir00 + '/test/pred'
        # ====for current test==================
        if config["dataset_name"]=="SRCD":
            config.test_dir = config.data_dir + '\\testCD'
            config.test_pred_dir = config.data_dir + '\\testCD\\pred'
        else:
            if config["network_G_CD"]["patch_size"] == 256:
                config.test_dir = config.data_dir + '\\test'
                config.test_pred_dir = config.data_dir + '\\test\\pred'
            else:
                config.test_dir = config.data_dir + '\\test'
                config.test_pred_dir = config.data_dir + '\\test\\pred'


            # config.test_dir = config.data_dir + '\\train'
            # config.test_pred_dir = config.data_dir + '\\train\\pred'


        mkdir_if_not_exist(config.test_pred_dir)

        device_ID='3090'

        if config["model"]=='MRCD':
            # ============for CD=============================
            src_tgt_name = 'Sense-CD'
            print(src_tgt_name)
            # config.pred_name = 'netG2_Res34Res_{}_diffmode_{}_dtype_{}_Drop_{:.2f}_ce_weight_{:.2f}_patch_{}_batch_{}_nepoch_{}_warmepoch_{}'.format(
            # config["network_G_CD"]["which_model_G"],config["network_G_CD"]["diff_mode"],config["network_G_CD"]["dblock_type"],config["train"]["drop_rate"],config["train"]["ce_weight"],
            # config.patch_size, config.batch_size, config["train"]["nepoch"],config["train"]["warmup_epoch"])#for sensetime-cd

            config.pred_name = 'netG3090V3_{}_diffmode_{}_dtype_{}_backbone_{}_patch_{}_batch_{}_nepoch_{}_warmepoch_{}_useDS_{}_useAtt_{}_useOnehotloss_{}'.format(
                config["network_G_CD"]["which_model_G"], config["network_G_CD"]["diff_mode"],
                config["network_G_CD"]["dblock_type"],
                #config["network_G_CD"]["ASPP_type"],
                config["network_G_CD"]["backbone"],
                config.patch_size, config.batch_size, config["train"]["nepoch"], config["train"]["warmup_epoch"],config["network_G_CD"]["use_DS"],
                config["network_G_CD"]["use_att"],config["train"]["use_onehot_loss"])#for whu-cd
        else:
            if config["network_G_CD"]["which_model_G"]=="CDFormer" or config["network_G_CD"]["which_model_G"]=="ScratFormer":
                config.pred_name ="netG{}_{}_img256_batch_{}_epoch_{}".format(device_ID,config["network_G_CD"]["which_model_G"],config.batch_size,config["train"]["nepoch"])
            elif config["network_G_CD"]["which_model_G"]=="CD_BIT" or config["network_G_CD"]["which_model_G"]=="MISSFormer" or config["network_G_CD"]["which_model_G"]=="UCTranSiam" or config["network_G_CD"]["which_model_G"]=="ICIFSiam"\
                    or config["network_G_CD"]["which_model_G"]=="CCLNet" or config["network_G_CD"]["which_model_G"]=="SARSNet":
                config.pred_name ="netG{}_data{}_{}_bone_{}_LR_{}_useCT_{}_useDecPos_{}_patch{}_batch_{}_epoch_{}".format(device_ID,config.dataset_name,config["network_G_CD"]["which_model_G"],config["network_G_CD"]["backbone"],config["train"]["lr_scheme"],config["network_G_CD"]["use_centerTrans"],config["network_G_CD"]["dec_pos"],config["network_G_CD"]["patch_size"],config.batch_size,config["train"]["nepoch"])

            elif config["network_G_CD"]["which_model_G"] == "UCTranSiam_Fuse" or config["network_G_CD"]["which_model_G"] == "UCTranSiam_Fuse2":
                config.pred_name = "netG{}_data_{}_{}_LR_{}_cosT_{}_SO_{}_fuse_{}_useMix_{}_att_{}_MLP_{}_LNum_{}_APE_{}_prjtype_{}_patch{}_batch_{}_epoch_{}_lr_{}_inter_{}_dec_{}_feaE_{}_upM_{}".format(
                    device_ID,config.dataset_name, config["network_G_CD"]["which_model_G"],
                    config["train"]["lr_scheme"],config["train"]["cosT"],config["network_G_CD"]["use_sideOut"],config["network_G_CD"]["fuse_mode"],config["network_G_CD"]["use_mix"],config["network_G_CD"]["fusion_type"],config["network_G_CD"]["mlp_type"],config["network_G_CD"]["LNum"],config["network_G_CD"]["use_APE"],config["network_G_CD"]["prj_type"],
                    config["network_G_CD"]["patch_size"], config.batch_size, config["train"]["nepoch"],config["train"]["lr_G"],config["network_G_CD"]["inter_mode"],
                    config["network_G_CD"]["dec_mode"], config["network_G_CD"]["feaE_mode"],config["network_G_CD"]["up_mode"])

            elif config["network_G_CD"]["which_model_G"] == "RCTransNet":
                config.pred_name = "netG{}_{}_{}_bone_{}_LR_{}_cosT_{}_SO_{}_fuse_{}_att_{}_patch{}_batch_{}_epoch_{}_lr_{}_OS_{}_inter_{}_dec_{}_feaE_{}_upM_{}".format(
                    device_ID, config.dataset_name, config["network_G_CD"]["which_model_G"],
                    config["network_G_CD"]["backbone"],
                    config["train"]["lr_scheme"], config["train"]["cosT"], config["network_G_CD"]["use_sideOut"],
                    config["network_G_CD"]["fuse_mode"],
                    config["network_G_CD"]["fusion_type"],
                    config["network_G_CD"]["patch_size"], config.batch_size, config["train"]["nepoch"],
                    config["train"]["lr_G"], config["network_G_CD"]["out_stride"], config["network_G_CD"]["inter_mode"],
                    config["network_G_CD"]["dec_mode"], config["network_G_CD"]["feaE_mode"],config["network_G_CD"]["up_mode"])



            elif config["network_G_CD"]["which_model_G"] == "UCTranSiam_CTrans":
                config.pred_name = "netG{}_data_{}_{}_bone_{}_LR_{}_cosT_{}_fuse_{}_att_{}_MLP_{}_LNum_{}_APE_{}_prjtype_{}_patch{}_batch_{}_epoch_{}_lr_{}".format(
                    device_ID,config.dataset_name, config["network_G_CD"]["which_model_G"], config["network_G_CD"]["back_mode"],
                    config["train"]["lr_scheme"],config["train"]["cosT"],config["network_G_CD"]["fuse_mode"],config["network_G_CD"]["fusion_type"],config["network_G_CD"]["mlp_type"],config["network_G_CD"]["LNum"],config["network_G_CD"]["use_APE"],config["network_G_CD"]["prj_type"],
                    config["network_G_CD"]["patch_size"], config.batch_size, config["train"]["nepoch"],config["train"]["lr_G"])


            elif config["network_G_CD"]["which_model_G"] == "MISSFormerSiam" or config["network_G_CD"]["which_model_G"] == "TopFormerSiam":
                config.pred_name = "netG{}SA_data{}_{}_bone_{}_decoder_{}_diffmode_{}_dangle_{}_patch{}_batch_{}_epoch_{}".format(
                    device_ID,
                    config.dataset_name, config["network_G_CD"]["which_model_G"], config["network_G_CD"]["backbone"],config["network_G_CD"]["decoder_type"],
                    config["network_G_CD"]["diff_mode"], config["network_G_CD"]["use_dangle"],config.patch_size, config.batch_size,
                    config["train"]["nepoch"])
            elif config["network_G_CD"]["which_model_G"]=="UTNet":
                config.pred_name ="netG{}_data{}_{}_patch{}_batch_{}_epoch_{}".format(device_ID,config.dataset_name,config["network_G_CD"]["which_model_G"],config.patch_size,config.batch_size,config["train"]["nepoch"])
            elif config["network_G_CD"]["which_model_G"]=="UNet_Trans" or config["network_G_CD"]["which_model_G"]=="DeepLabCD0" or config["network_G_CD"]["which_model_G"]=="DeepLabCD":
                config.pred_name ="netG{}_{}_semiMod_{}_LR_{}_batch_{}_epoch_{}_semi_{}_{}_th_{}_lam_{}_unw_{}".format(device_ID,config["network_G_CD"]["which_model_G"],config["train"]["semi_mode"],config['train']['lr_scheme'],config.batch_size,config["train"]["nepoch"],config["train"]["semi_ratio"],config.dataset_name+'2'+config.dataset_name2,config["train"]["conf_thresh"],config["train"]["lamda_u"],config["train"]["lamda_u"])
            elif config["network_G_CD"]["which_model_G"]=="UNet_MLP" or config["network_G_CD"]["which_model_G"]=="A2Net":
                config.pred_name ="netG{}_{}_img256_nf_{}_batch_{}_epoch_{}".format(device_ID,config["network_G_CD"]["which_model_G"],config["network_G_CD"]["nf"],config.batch_size,config["train"]["nepoch"])
            else:



                raise NotImplementedError('Model [{:s}] not recognized.'.format(config["network_G_CD"]["which_model_G"]))


                #=============================for SSCD=====================================

        if config.mode=="Test":

           if config["model"] == 'MRCD':
               if config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU" or config["network_G_CD"]["which_model_G"]=="FC_EF" or config["network_G_CD"]["which_model_G"]=="Seg_EF"or config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN" or config["network_G_CD"]["which_model_G"]=="EDCls_UNet_BCD_WHU_DCN2":
                   config.pred_dir=config.test_pred_dir +'/'+config.pred_name
                   mkdir_if_not_exist(config.pred_dir)
                   config.precision_path = config.pred_dir + '/acc.txt'
               else:
                   if config.use_CRF:
                       config.pred1_dir = config.test_pred_dir + '\\im1_gray_tta_crf_' + config.pred_name
                       config.pred2_dir = config.test_pred_dir + '\\im2_gray_tta_crf_' + config.pred_name
                       config.pred1_rgb_dir = config.test_pred_dir + '\\im1_rgb_tta_crf_' + config.pred_name
                       config.pred2_rgb_dir = config.test_pred_dir + '\\im2_rgb_tta_crf_' + config.pred_name
                   else:
                       config.pred1_dir = config.test_pred_dir + '\\im1_gray_tta_' + config.pred_name
                       config.pred2_dir = config.test_pred_dir + '\\im2_gray_tta_' + config.pred_name
                       config.pred1_rgb_dir = config.test_pred_dir + '\\im1_rgb_tta_' + config.pred_name
                       config.pred2_rgb_dir = config.test_pred_dir + '\\im2_rgb_tta_' + config.pred_name
                   mkdir_if_not_exist(config.pred1_dir)
                   mkdir_if_not_exist(config.pred2_dir)
                   mkdir_if_not_exist(config.pred1_rgb_dir)
                   mkdir_if_not_exist(config.pred2_rgb_dir)
                   config.precision_path = config.pred1_dir + '/acc.txt'


           else:


               config.pred_dir = config.test_pred_dir + '/pred_' + config.pred_name
               mkdir_if_not_exist(config.pred_dir)
               mkdir_if_not_exist(config.pred_dir+'/Binary')
               config.precision_path = config.pred_dir + '/acc.txt'


        print("pred_model is {}".format(config.pred_name))





        #==============for model======================

        config.model_name = config.model_dir + '/'+config.pred_name+'.pth'

        #==============for log=========================
        config.json_name = config.model_dir + '/min_max.json'
        config.loss_path = config.img_dir + '/' + config.pred_name + '.png'
        config.log_path = config.model_dir + '/' + config.pred_name + '.txt'
        # #============for ramdom mean std===============
        # config.meanA=config.data_dir+'/train/meanA.npy'
        # config.stdA=config.data_dir+'/train/stdA.npy'
        # config.meanB = config.data_dir + '/train/meanB.npy'
        # config.stdB = config.data_dir + '/train/stdB.npy'
        #=============================================
        mkdir_if_not_exist(config.model_dir)
        mkdir_if_not_exist(config.img_dir)
        mkdir_if_not_exist(config.log_dir)
        #===================================================

        self.config=config
        self.data_dir=config.data_dir
        self.train_dir=config.train_dir
        self.test_dir = config.test_dir
        self.val_dir=config.data_dir+'/val'
        self.json_name=config.json_name



    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


    def write_json(self,data=[]):
        f=open(self.json_name,"w")
        result=[]
        temp={}
        temp['mean']=data[0]
        temp['std']=data[1]
        result.append(temp)
        f.write(json.dumps(result,cls=MyEncoder))










