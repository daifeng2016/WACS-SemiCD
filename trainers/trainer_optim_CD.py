import math
import os, time,sys
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loaders.RSCD_dl import RSCD_DL
#from torchsummary import summary
import matplotlib.pyplot as plt

import logging
from losses.myLoss import bce_edge_loss
#from losses.dist_angle import DAngleLoss
from utils.utils import PR_score_whole

from tqdm import tqdm
import random
from losses.myLoss import entropy_loss
import matplotlib
matplotlib.use('TkAgg')

class TrainerOptim(object):
    # init function for class
    def __init__(self, config,trainDataloader, valDataloader,trainDataloader2=None
                ):
        dl=RSCD_DL(config)
        self.config=dl.config
        self.model_path=dl.config.model_name
        self.log_file=dl.config.log_path
        self.lossImg_path=dl.config.loss_path

        # set the GPU flag

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.valDataloader = valDataloader
        self.trainDataloader = trainDataloader
        self.trainDataloader2 = trainDataloader2

        self.pix_cri=bce_edge_loss(use_edge=False).to(self.device)
        self.pix_cri0 = bce_edge_loss(use_edge=False).to(self.device)
        self.cri_dangle=DAngleLoss().to(self.device)
        from models.Satt_CD.modules.loss import BCL
        self.cri_dist = BCL().to(self.device)
        self.use_dangle=self.config["network_G_CD"]["use_dangle"]


    def load_ck(self,model,model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_dict = checkpoint
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)


    def train_optim(self):
        # create model
        start_time = time.perf_counter()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger
        #model = create_model(self.config)
        # resume state??
        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"iter_epoch":[],
            "loss": [],
                         "acc":[],
                         "val_loss": [],
                         "val_acc": []
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter']=total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        #multi_outputs = self.config["network_G-CD"]["multi_outputs"]
        best_acc = 0
        best_loss=1000
        # if self.config["network_G_CD"]["which_model_G"]=='EDCls_Net':
        #    self.load_ck(model.netG,self.config.pretrained_model_path)#loading model checkpoint
        if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':

           self.load_ck(model.netCD, self.config.pretrained_model_path)  # loading model checkpoint
           model.netCD.eval()

        use_warmup=True if self.config["train"]["warmup_epoch"]>0 else False
        if use_warmup:
            logger.info("using warmup for training")
            for optim in model.optimizers:
                optim.zero_grad()
                optim.step()
        if self.config["train"]["use_progressive_resize"]:
            logger.info("using progressive resize for training")


        if self.config.use_KFold:
            model.reset_models()


        for epoch in range(0, total_epochs):
            print('Epoch {}/{}'.format(epoch + 1, total_epochs))
            print('-' * 60)
            epoch_loss = 0
            epoch_acc = 0
            train_history["iter_epoch"].append(epoch)

            if use_warmup or self.config["train"]["lr_scheme"]=="CosineLR" or self.config["train"]["lr_scheme"]=="PolyCycLR" or self.config["train"]["lr_scheme"]=="CosineRe":
                model.update_learning_rate()
            cur_Dataloader = self.trainDataloader

            for i, sample in enumerate(tqdm(cur_Dataloader, 0)):
                current_step += 1
                # training
                model.feed_data(sample)
                model.netG.train()

                #model.optimize_parameters(current_step)
                if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':#for ISCD

                    model.optimize_parameters_MC7_rgb255(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_Res50':

                    model.optimize_parameters_MC7_DS(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_DCN'\
                        or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_DCN2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_Flow' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny_Siam' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_CAM_Siam'\
                        or self.config["network_G_CD"]["which_model_G"] == 'Tiny_CD' or self.config["network_G_CD"]["which_model_G"] == 'CTFINet':
                    model.optimize_parameters_SC2(current_step)#double inputs CD
                elif self.config['network_G_CD']['which_model_G'] == 'FC_EF' or self.config["network_G_CD"]["which_model_G"]=="Seg_EF" or self.config["network_G_CD"]["which_model_G"]=="UNetPlusPlus":
                    model.optimize_parameters_SC1(current_step)#single input  CD
                elif self.config['network_G_CD']['which_model_G'] == 'CDFormer' or self.config['network_G_CD']['which_model_G'] =='CD_BIT' or \
                        self.config['network_G_CD']['which_model_G'] =='UNet_Trans'or self.config['network_G_CD']['which_model_G'] =='UNet_MLP'\
                        or self.config['network_G_CD']['which_model_G'] =='ScratFormer'or self.config['network_G_CD']['which_model_G'] =='MISSFormer'\
                        or self.config["network_G_CD"]["which_model_G"]=="MISSFormerSiam" or self.config["network_G_CD"]["which_model_G"]=="TopFormerSiam"\
                        or self.config["network_G_CD"]["which_model_G"]=="UCTranSiam" or self.config["network_G_CD"]["which_model_G"]=="ICIFSiam"\
                        or self.config["network_G_CD"]["which_model_G"]=="UCTranSiam_Fuse" or self.config["network_G_CD"]["which_model_G"]=="UCTranSiam_Fuse2"\
                        or self.config["network_G_CD"]["which_model_G"]=="RCTransNet" or self.config["network_G_CD"]["which_model_G"] == 'CCLNet'  \
                        or self.config["network_G_CD"]["which_model_G"] == 'SARSNet' or self.config["network_G_CD"]["which_model_G"]=="UCTranSiam_CTrans":
                    model.optimize_parameters_CDFormer(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                    model.optimize_parameters_SCSeg(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_DSCD':
                    model.optimize_parameters_MC(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin'or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin':
                    model.optimize_parameters_MC7_bin(current_step)
                elif self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                    model.optimize_parameters_MC6_bin(current_step)
                else:
                    if self.config["train"]["use_CatOut"]:
                        model.optimize_parameters_MC7_DS(current_step)
                    else:
                        model.optimize_parameters_MC7(current_step)



                # update learning rate
                if use_warmup==False and self.config["train"]["lr_scheme"]=="MultiStepLR":
                   model.update_learning_rate()

                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    epoch_loss += logs['l_g_total']
                    #epoch_acc += logs['pnsr']
                    # message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}, lossD_grad:{:.6f}> '.format(
                    #     epoch, current_step, model.get_current_learning_rate(),
                    #     logs['l_g_total'],
                    #     0.5*(logs['l_d_real']+logs['l_d_fake']),0.5*(logs['l_d_real_grad']+logs['l_d_fake_grad']))

                    # message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                    #     epoch, current_step, model.get_current_learning_rate(use_warmup=use_warmup),
                    #     logs['l_g_total'],logs['l_d_total']
                    #
                    # )
                    if self.config.use_KFold:
                        message = '<fold:{:2d}, epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}>'.format(
                            self.config.fold_index, epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                            logs['l_g_total'], logs['l_d_total'])
                    else:
                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                        logs['l_g_total'],logs['l_d_total'])

                    logger.info(message)
                    #=======for val test======================================
                    # val_loss, val_acc = self.val_CDFormer(epoch + 1, model=model)
                    # # # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    # # # #val_loss, val_acc = self.val(epoch + 1, model=model)
                    # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    # logger.info(message)
                    #=========================================================
                if current_step in self.config['logger']['save_iter']:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
            if epoch % self.config['train']['val_epoch'] == 0:
                # if self.config["network_G_CD"]["which_model_G"] == 'EDCls_Net2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet' or \
                #     self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet4'\
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_DiffAdd' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet5' \
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5'\
                #         or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_Res50':
                #     val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                # else:
                #     val_loss, val_acc = self.val(epoch + 1, model=model, multi_outputs=True)
                if self.config["network_G_CD"]["out_nc"]>1:
                    if self.config["network_G_CD"]["out_nc"]==2:
                        #val_loss, val_acc = self.val_SEK32(epoch + 1, model=model)
                        val_loss, val_acc =self.val_CDFormer(epoch + 1, model=model)
                    else:
                        val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)

                else:
                    val_loss, val_acc = self.val(epoch + 1, model=model)

                message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                logger.info(message)
                logs = model.get_current_log()
                #train_history["loss"].append(logs['l_g_total'])
                train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                #train_history["acc"].append(epoch_acc * 1.0 / len(self.trainDataloader))
                train_history["val_loss"].append(val_loss)
                train_history["val_acc"].append(val_acc)
                if self.config["network_G_CD"]["out_nc"] < 10:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        model.save_best_acc()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        model.save_best_loss()



        end_time = time.perf_counter()
        run_time=end_time-start_time
        #print(end_time - start_time, 'seconds')
        message='running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim2(train_history)

    def train_optim_tune(self):

        # create model
        start_time = time.clock()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger

        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc":[],
                         "val_loss": [],
                         "val_acc": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter']=total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        #multi_outputs = self.config["network_G-CD"]["multi_outputs"]
        best_acc = 0
        best_loss=1000

        self.load_ck(model.netG,self.config.pretrained_model_path)#loading model checkpoint
        use_warmup = True if self.config["train"]["warmup_epoch"] > 0 else False
        #=======frozon feature param=======
        for p in model.netG.feat_Extactor.parameters():
            p.requires_grad = False

        # use_warmup=True if self.config["train"]["warmup_epoch"]>0 else False
        # if use_warmup:
        #     logger.info("using warmup for training")
        #     for optim in model.optimizers:
        #         optim.zero_grad()
        #         optim.step()

        logger.info("using fine-tune for training,frozen the params of the feat_extrator...")
        for epoch in range(0, total_epochs):
            print('Epoch {}/{}'.format(epoch + 1, total_epochs))
            print('-' * 60)
            epoch_loss = 0
            epoch_acc = 0

            # if use_warmup:
            #     model.update_learning_rate()
            if use_warmup or self.config["train"]["lr_scheme"]=="CosineLR":
                model.update_learning_rate()#update lr pr epoch for consineLR




            cur_Dataloader = self.trainDataloader

            for i, sample in enumerate(tqdm(cur_Dataloader, 0)):
                current_step += 1
                # training
                model.feed_data(sample)
                model.netG.train()
                model.optimize_parameters_MC7(current_step)


                # update learning rate
                #if use_warmup==False:
                #model.update_learning_rate()
                if use_warmup==False and self.config["train"]["lr_scheme"]=="MultiStepLR":
                   model.update_learning_rate()

                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    epoch_loss += logs['l_g_total']
                    if self.config.use_KFold:
                        message = '<fold:{:2d}, epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}>'.format(
                            self.config.fold_index, epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                            logs['l_g_total'], logs['l_d_total'])
                    else:
                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                        logs['l_g_total'],logs['l_d_total'])


                    logger.info(message)
                    #=======for val test======================================
                    # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    # logger.info(message)
                    #=========================================================
                if current_step in self.config['logger']['save_iter']:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
            if epoch % self.config['train']['val_epoch'] == 0:

                val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)

                message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                logger.info(message)
                logs = model.get_current_log()
                #train_history["loss"].append(logs['l_g_total'])
                train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                #train_history["acc"].append(epoch_acc * 1.0 / len(self.trainDataloader))
                train_history["val_loss"].append(val_loss)
                train_history["val_acc"].append(val_acc)
                if val_acc> best_acc:
                    best_acc = val_acc
                    model.save_best_acc()
                if val_loss<best_loss:
                    best_loss=val_loss
                    model.save_best_loss()



        end_time = time.clock()
        run_time=end_time-start_time
        #print(end_time - start_time, 'seconds')
        message='running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim(train_history)



    def train_optim_cos(self):
        # create model
        start_time = time.clock()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger
        # model = create_model(self.config)
        # resume state??
        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc": [],
                         "val_loss": [],
                         "val_acc": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader))
        self.config['train']['niter'] = total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        multi_outputs = self.config["network_G"]["multi_outputs"]
        best_acc = 0
        best_loss = 1000
        if self.config["network_G_CD"] == 'EDCls_Net':
            self.load_ck(model.netG, self.config.pretrained_model_path)  # loading model checkpoint

        init_lr = self.config["train"]["lr_G"]
        epochs_per_cycle = total_epochs // self.config["train"]["cos_cycle"]
        model_shots = []
        for i_cycle in range(self.config["train"]["cos_cycle"]):
            for epoch in range(epochs_per_cycle):
                print('Cycle_{},Epoch {}/{}'.format(i_cycle,epoch + 1, epochs_per_cycle))
                print('-' * 60)
                epoch_loss = 0
                epoch_acc = 0
                cur_lr=model.update_learning_rate_cos(init_lr, epoch, epochs_per_cycle)

                for i, sample in enumerate(tqdm(self.trainDataloader, 0)):
                    current_step += 1
                    # training
                    model.feed_data(sample)
                    model.netG.train()
                    model.optimize_parameters_MC7(current_step)

                    if current_step % self.config['logger']['print_freq'] == 0:
                        logs = model.get_current_log()
                        epoch_loss += logs['l_g_total']

                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                            epoch, current_step, cur_lr,
                            logs['l_g_total'], logs['l_d_total']
                        )
                        logger.info(message)
                        # =======for val test======================================
                        # val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                        # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                        # logger.info(message)
                        # =========================================================
                    # if current_step in self.config['logger']['save_iter']:
                    #     logger.info('Saving models and training states.')
                    #     model.save(current_step)

                if epoch % self.config['train']['val_epoch'] == 0:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_Net2' or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet' or \
                            self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2' or \
                            self.config["network_G_CD"][
                                "which_model_G"] == 'EDCls_UNet3' or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet4'or self.config["network_G_CD"][
                        "which_model_G"] == 'EDCls_UNet2_DiffAdd':
                        val_loss, val_acc = self.val_SEK7(epoch + 1, model=model)
                    else:
                        val_loss, val_acc = self.val(epoch + 1, model=model, multi_outputs=multi_outputs)

                    message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                    logger.info(message)
                    logs = model.get_current_log()
                    train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader))
                    train_history["val_loss"].append(val_loss)
                    train_history["val_acc"].append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        model.save_best_acc_cycle(i_cycle)


        end_time = time.clock()
        run_time = end_time - start_time
        message = 'running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim(train_history)




    def visualize_train_optim(self, history):

        val_acc = history["val_acc"]
        loss = history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        plt.subplot(121)
        #plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        #epoch_array=np.arrange(len(val_acc))
        plt.plot(val_acc)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')


        plt.legend(['train', 'valid'], loc='upper right')
        if self.config.use_KFold:
            lossImg_path=self.config.img_dir + '/' + self.config.pred_name +'_fold'+str(self.config.fold_index)+'.png'
            plt.savefig(lossImg_path)
            plt.close()# close the window before drawing another curve
        else:
           plt.savefig(self.lossImg_path)
           plt.show()

    def visualize_train_optim2(self, history):
        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(hspace=0.3)
        val_acc = history["val_acc"]
        loss = history["loss"]
        acc = history["acc"]
        val_loss = history["val_loss"]

        plt.subplot(121)
        #https://blog.csdn.net/sinat_36219858/article/details/79800460
        # plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        epoch_array=np.arange(len(val_acc))
        plt.plot(epoch_array,val_acc,'r^-')
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(epoch_array,loss,'b*-')
        plt.plot(epoch_array,val_loss,'yo-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.legend(['train', 'valid'], loc='upper right')
        if self.config.use_KFold:
            lossImg_path = self.config.img_dir + '/' + self.config.pred_name + '_fold' + str(
                self.config.fold_index) + '.png'
            plt.savefig(lossImg_path)
            plt.close()  # close the window before drawing another curve
        else:

            plt.savefig(self.lossImg_path)
            plt.show()

            #self.config.model_dir + '/' + self.config.pred_name + '_best_loss.pth'
            save_path=self.config.log_dir + '/' + self.config.pred_name + '_curve'+'.txt'
            from data_loaders.RSCD_dl import MyEncoder
            import json
            f = open(save_path, "w")
            # result = []
            # temp = {}
            # temp['mean'] = data[0]
            # temp['std'] = data[1]
            # result.append(temp)
            f.write(json.dumps(history, cls=MyEncoder))









    def visualize_train(self,history):
        acc = history["acc"]
        val_acc = history["val_acc"]
        loss = history["loss"]
        val_loss = history["val_loss"]
        plt.subplot(121)
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model pnsr')
        plt.ylabel('pnsr')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')

        plt.savefig(self.lossImg_path)
        plt.show()

    def print_info(self,history={},elapse_time=0.0,epochs=20):
        mylog = open(self.log_file, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog  # 输出到文件


        print(summary(self.net,(3, 48, 48)))

        print("model train time is %.6f s" % elapse_time)
        print('model_name:', self.model_path)
        loss=history['loss']# equal to history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        val_acc = history["val_acc"]
        for i in range(epochs):
            print('epoch: %d' % (i + 1))
            print('train_loss: %.5f' % loss[i], 'val_loss:%.5f' % val_loss[i])
            print('train_acc:%.5f' % acc[i], 'val_acc:%.5f' % val_acc[i])
            mylog.flush()
        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

    def save_checkpoint(self,state, is_best, filename=None):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print("==> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print("==> Validation Accuracy did not improve")


    def  val_SEK32(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set

        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        infer_list=[]
        label_list=[]
        # perform test inference
        val_model = model.netG
        val_model.eval()

        change_type = ['0_0',
                       '1_2', '1_3', '1_4', '1_5', '1_6',
                       '2_1', '2_3', '2_4', '2_5', '2_6',
                       '3_1', '3_2', '3_4', '3_5', '3_6',
                       '4_1', '4_2', '4_3', '4_5', '4_6',
                       '5_1', '5_2', '5_3', '5_4', '5_5', '5_6',
                       '6_1', '6_2', '6_3', '6_4', '6_5']

        rgb_table = {'0': (255, 255, 255), '1': (0, 0, 255), '2': (128, 128, 128), '3': (0, 128, 0),
                     '4': (0, 255, 0), '5': (128, 0, 0), '6': (255, 0, 0)}
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()

                #outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                outputs = val_model(imgs)
                outputs0 = torch.argmax(outputs, dim=1)#[4,32,256,256]==>[4,256,256]
                outputs0 = outputs0.data.cpu().numpy().astype('uint8')

                #batch_size=masks_pred.shape[0]
                for b in range(outputs0.shape[0]):
                    masks_pred=outputs0[b,...]
                    masks_label=labels[b,...]
                    # pred1 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    # pred2 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    #
                    # label1 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')
                    # label2 = np.zeros((masks_pred.shape + (3,)), dtype='uint8')

                    pred1 = np.zeros((masks_pred.shape), dtype='uint8')
                    pred2 = np.zeros((masks_pred.shape), dtype='uint8')

                    label1 = np.zeros((masks_pred.shape), dtype='uint8')
                    label2 = np.zeros((masks_pred.shape), dtype='uint8')

                    for i in range(masks_pred.shape[0]):
                        for j in range(masks_pred.shape[1]):
                            cur_change = change_type[masks_pred[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]
                            # key1 = str(idx1)
                            # key2 = str(idx2)
                            pred1[i,j]=idx1
                            pred2[i,j]=idx2

                            cur_change = change_type[masks_label[i, j]]
                            idx1 = cur_change[:cur_change.find('_')]
                            idx2 = cur_change[cur_change.find('_') + 1:]
                            label1[i,j]=idx1
                            label2[i,j]=idx2

                            # key1_label = str(idx1)
                            # key2_label = str(idx2)

                            #for k in range(3):
                                # pred1[i, j, k] = rgb_table[key1][k]
                                # pred2[i, j, k] = rgb_table[key2][k]
                                # label1[i, j, k] = rgb_table[key1_label][k]
                                # label2[i, j, k] = rgb_table[key2_label][k]

                    infer_list.append(pred1)
                    infer_list.append(pred2)
                    label_list.append(label1)
                    label_list.append(label2)
                    del pred1,pred2,label1,label2




                ce_loss = nn.CrossEntropyLoss()
                loss = ce_loss(outputs, labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()


        from utils.SCDD_eval import Eval_preds
        _,_,score=Eval_preds(infer_list,label_list)

        val_loss=lossAcc*1.0/len(self.valDataloader)
        val_acc=score

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training



        return  val_loss,val_acc



    def  val_SEK7(self, epoch,model=None):
        # eval model on validation set

        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        infer_list=[]
        label_list=[]
        # perform test inference
        val_model = model.netG
        val_model.eval()

        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                gt1_label=labels[:,0,:,:].data.cpu().numpy().astype('uint8')
                gt2_label = labels[:, 1, :, :].data.cpu().numpy().astype('uint8')

                if self.config["network_G_CD"]["use_DS"]:
                    #if self.config["train"]["use_label_rgb255"]:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':#for rgb255 guidance
                        pred1, pred2, pred3, (outputs1, outputs2,_) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    #elif  self.config["train"]["use_CatOut"]:
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                        pred1, pred2, pred3,pred4, (outputs1, outputs2) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                        model.netCD.eval()
                        images_T1, images_T2 = imgs[:, 0:3, ...], imgs[:, 3:6, ...]
                        with torch.no_grad():
                            _, _, _, images_cd = model.netCD(images_T1, images_T2)

                        images_T1 = torch.cat([images_T1, images_cd], dim=1)
                        images_T2 = torch.cat([images_T2, images_cd], dim=1)
                        outputs1, outputs2 = val_model(images_T1, images_T2)
                    elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin':
                        pred1, pred2, pred3, (outputs1, outputs2,outputs12) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif  self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                        outputs1, outputs2, outputs12 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        #_, _, _, (outputs1, outputs2) = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                        pred1,pred2,pred3,(outputs1, outputs2)= val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                else:
                    if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet3':
                        outputs1, outputs2,_ = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    elif  self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                        outputs1, outputs2, outputs12 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        outputs1, outputs2 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                outputs1_label = torch.argmax(outputs1, dim=1)#[4,32,256,256]==>[4,256,256]
                outputs1_label = outputs1_label.data.cpu().numpy().astype('uint8')

                outputs2_label=torch.argmax(outputs2, dim=1)
                outputs2_label = outputs2_label.data.cpu().numpy().astype('uint8')

                if self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC7Bin' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_MC6Bin' or \
                        self.config["network_G_CD"]["which_model_G"] == 'DeepLab_SCD' or self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                    outputs12 = outputs12[:, 0, :, :].data.cpu().numpy()
                    if self.config["train"]["use_MC6"]:
                        outputs1_label+=1
                        outputs2_label += 1
                        outputs1_label[outputs12 < 0.5] = 0
                        outputs2_label[outputs12 < 0.5] = 0
                    else:

                        outputs1_label[outputs12<0.5]=0
                        outputs2_label[outputs12<0.5]=0


                infer_list.append(outputs1_label)
                infer_list.append(outputs2_label)
                label_list.append(gt1_label)
                label_list.append(gt2_label)


                #====for loss=============
                if self.config["train"]["use_CatOut"] or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet2_New5':
                    loss = self.compute_val_loss_Cat(pred1, pred2, pred3,pred4, (outputs1, outputs2), labels)
                elif self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_Seg':
                    loss=self.compute_val_loss_single(outputs1,outputs2,labels)
                elif self.config["network_G_CD"]["which_model_G"] == 'HRNet_SCD':
                    # l_g_total = 0
                    labels -= 1
                    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
                    loss = ce_loss(outputs1, labels[:, 0, ...]) + ce_loss(outputs2, labels[:, 1, ...])
                else:
                    #loss=self.compute_val_loss(pred1,pred2,pred3,(outputs1,outputs2),labels)
                    if self.config["network_G_CD"]["use_DS"]:
                        loss=self.compute_val_loss(pred1,pred2,pred3,(outputs1,outputs2),labels)
                    else:
                        loss = self.compute_val_loss_single(outputs1, outputs2, labels)




                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()


        from utils.SCDD_eval import Eval_preds
        _,_,score=Eval_preds(infer_list,label_list)

        val_loss=lossAcc*1.0/len(self.valDataloader)
        val_acc=score

        print('Epoch %d evaluate done ' % epoch)




        return  val_loss,val_acc

    def compute_val_loss(self,pred1,pred2,pred3,pred4,labels):

        # # =========================for sensetime cd==========================================================
        # # class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        # # class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        #
        # # ======================for franch cd====================
        #
        # class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
        # class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]

        if self.config["dataset_name"] == 'sensetime':
            class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
            class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]


        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]

        if self.config["network_G_CD"]["use_DS"]:

            if self.config.patch_size == 256:
                img_down_size = 16
            else:
                img_down_size = 32
            if self.config["train"]["use_progressive_resize"]:
                img_down_size = 16

            labels1 = F.interpolate(labels.float(), (img_down_size, img_down_size), mode='bilinear', align_corners=True)
            labels2 = F.interpolate(labels.float(), (img_down_size * 2, img_down_size * 2), mode='bilinear',
                                    align_corners=True)
            labels3 = F.interpolate(labels.float(), (img_down_size * 4, img_down_size * 4), mode='bilinear',
                                    align_corners=True)
            labels1, labels2, labels3= labels1.long(), labels2.long(), labels3.long()

            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels4_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels4_2[:, k, ...])

            ce_weight = self.config["train"]["ce_weight"]
            l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                       labels1[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                       labels2[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                       labels3[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred4[0], labels[:, 0, ...]) * ce_weight*2 + self.cri_ce_loss(pred4[1],
                                                                                                      labels[:, 1,
                                                                                                      ...]) * ce_weight*2
        return l_g_total

    def compute_val_loss_Cat(self,pred1,pred2,pred3,pred4,pred5,labels):
        class_weight1 = [0.0007, 0.1970, 0.0065, 0.0085, 0.0391, 0.0176, 0.7307]
        class_weight2 = [0.0011, 0.3595, 0.0154, 0.0265, 0.0524, 0.0104, 0.5347]
        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]

        if self.config["train"]["use_DS"]:

            if self.config.patch_size == 256:
                img_down_size = 16
            else:
                img_down_size = 32
            if self.config["train"]["use_progressive_resize"]:
                img_down_size = 16

            labels1 = F.interpolate(labels.float(), (img_down_size, img_down_size), mode='bilinear', align_corners=True)
            labels2 = F.interpolate(labels.float(), (img_down_size * 2, img_down_size * 2), mode='bilinear',
                                    align_corners=True)
            labels3 = F.interpolate(labels.float(), (img_down_size * 4, img_down_size * 4), mode='bilinear',
                                    align_corners=True)
            labels4 = F.interpolate(labels.float(), (img_down_size * 8, img_down_size * 8), mode='bilinear',
                                    align_corners=True)
            labels1, labels2, labels3,labels4= labels1.long(), labels2.long(), labels3.long(),labels4.long()

            one_hot_labels1_1 = one_hot_cuda(labels1[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels1_2 = one_hot_cuda(labels1[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_1 = one_hot_cuda(labels2[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels2_2 = one_hot_cuda(labels2[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_1 = one_hot_cuda(labels3[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels3_2 = one_hot_cuda(labels3[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_1 = one_hot_cuda(labels4[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels4_2 = one_hot_cuda(labels4[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels5_1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
            one_hot_labels5_2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)

            for k in range(class_num):
                l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[0][:, k, ...], one_hot_labels1_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred1[1][:, k, ...], one_hot_labels1_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred2[0][:, k, ...], one_hot_labels2_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[1][:, k, ...], one_hot_labels2_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred3[0][:, k, ...], one_hot_labels3_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred3[1][:, k, ...], one_hot_labels3_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred4[0][:, k, ...], one_hot_labels4_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred4[1][:, k, ...], one_hot_labels4_2[:, k, ...])

                l_g_total += class_weight1[k] * self.cri_seg_mc(pred5[0][:, k, ...], one_hot_labels5_1[:, k, ...])
                l_g_total += class_weight2[k] * self.cri_seg_mc(pred5[1][:, k, ...], one_hot_labels5_2[:, k, ...])

            ce_weight = 4.0
            l_g_total += self.cri_ce_loss(pred1[0], labels1[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred1[1],
                                                                                                       labels1[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred2[0], labels2[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2[1],
                                                                                                       labels2[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred3[0], labels3[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred3[1],
                                                                                                       labels3[:, 1,
                                                                                                       ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred4[0], labels4[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred4[1],
                                                                                                      labels4[:, 1,
                                                                                                      ...]) * ce_weight
            l_g_total += self.cri_ce_loss(pred5[0], labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred5[1],
                                                                                                      labels[:, 1,
                                                                                                      ...]) * ce_weight
        return l_g_total

    def compute_val_loss_single(self,pred1,pred2,labels):
        if self.config["dataset_name"]=='sensetime':
            class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
            class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        else:
            class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
            class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]
        l_g_total=0
        from models.utils import one_hot_cuda
        from models.Satt_CD.modules.loss import ComboLoss
        self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss().to(self.device)
        class_num = self.config["network_G_CD"]["out_nc"]
        label_smooth = self.config["train"]["use_label_smooth"]
        one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
        ce_weight = self.config["train"]["ce_weight"]
        for k in range(class_num):
            l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
            l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2,
                                                                                               labels[:, 1,
                                                                                               ...]) * ce_weight

        return l_g_total

    def compute_val_loss_single2(self,pred1,pred2,labels):
        # if self.config["dataset_name"]=='sensetime':
        #     class_weight1 = [ 0.0007,    0.1970,    0.0065,    0.0085,    0.0391,    0.0176,    0.7307]
        #     class_weight2 = [0.0011,    0.3595,    0.0154,    0.0265,    0.0524,    0.0104,    0.5347]
        # else:
        #     class_weight1 = [0.0001, 0.3054, 0.0054, 0.2481, 0, 0.4410]
        #     class_weight2 = [0.0001, 0.0097, 0.0458, 0.7787, 0.1557, 0.0100]
        # l_g_total=0
        # from models.utils import one_hot_cuda
        # from models.Satt_CD.modules.loss import ComboLoss
        # self.cri_seg_mc = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).to(self.device)
        self.cri_ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        # class_num = self.config["network_G_CD"]["out_nc"]
        # label_smooth = self.config["train"]["use_label_smooth"]
        # one_hot_labels1 = one_hot_cuda(labels[:, 0, ...], num_classes=class_num, label_smooth=label_smooth)
        # one_hot_labels2 = one_hot_cuda(labels[:, 1, ...], num_classes=class_num, label_smooth=label_smooth)
        # ce_weight = self.config["train"]["ce_weight"]
        # for k in range(class_num):
        #     l_g_total += class_weight1[k] * self.cri_seg_mc(pred1[:, k, ...], one_hot_labels1[:, k, ...])
        #     l_g_total += class_weight2[k] * self.cri_seg_mc(pred2[:, k, ...], one_hot_labels2[:, k, ...])
        # l_g_total += self.cri_ce_loss(pred1, labels[:, 0, ...]) * ce_weight + self.cri_ce_loss(pred2,
        #                                                                                        labels[:, 1,
        #
        #
        #                                                                                        ...]) * ce_weight
        l_g_total=0
        labels-=1
        l_g_total+=self.cri_ce_loss(pred1, labels[:, 0, ...])+self.cri_ce_loss(pred2, labels[:, 1, ...])



        return l_g_total

    def  val(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.config['network_G_CD']['which_model_G'] == 'Feat_Cmp':
                    labels0 = F.interpolate(labels, size=torch.Size(
                        [imgs.shape[2] // self.config.ds, imgs.shape[3] // self.config.ds]), mode='nearest')
                    labels0[labels == 1] = -1  # change
                    labels0[labels == 0] = 1  # must convert ot [-1,1] before calculating  loss

                    featT1, featT2 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    dist = F.pairwise_distance(featT1, featT2, keepdim=True)
                    dist = F.interpolate(dist, size=imgs.shape[2:], mode='bilinear', align_corners=True)

                    outputs = (dist > 1).float()
                    loss = self.cri_dist(dist, labels0)
                elif self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD':
                    _, _, _, outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU' or self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU_DCN' or self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU_DCN2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_Flow' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny' \
                        or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny_Siam' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_CAM_Siam' or self.config["network_G_CD"]["which_model_G"] == 'CD_BIT'\
                        or self.config["network_G_CD"]["which_model_G"] == 'Tiny_CD' or self.config["network_G_CD"]["which_model_G"] == 'CTFINet' or self.config["network_G_CD"]["which_model_G"] == 'SARSNet'\
                        or self.config["network_G_CD"]["which_model_G"] == 'CCLNet':
                    if self.config["network_G_CD"]["use_DS"]:
                       _, _, _, outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                    if self.config['network_G_CD']['out_nc']==2:
                        preds_prob = F.softmax(outputs, dim=1)
                        outputs = preds_prob[:, 1].unsqueeze(1).float()

                    bce_loss = bce_edge_loss(use_edge=False).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif self.config['network_G_CD']['which_model_G'] == 'FC_EF' or self.config['network_G_CD']['which_model_G'] == 'Seg_EF' or self.config['network_G_CD']['which_model_G'] == 'UNet_MLP' or self.config['network_G_CD']['which_model_G'] == 'UNetPlusPlus':
                    outputs = val_model(imgs)
                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif  self.config['network_G_CD']['which_model_G'] == 'UNet_Trans':
                    outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                else:
                    _,_,_,outputs=val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    ce_loss=nn.CrossEntropyLoss()
                    loss=ce_loss(outputs,labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc

    def  val_CDFormer(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.config["network_G_CD"]["in_c"]==3:
                    if self.use_dangle:
                        preds,pred_mask,pred_dir=val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        preds = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                else:
                    preds = val_model(imgs)

                if isinstance(preds,list) or isinstance(preds,tuple):
                    loss = 0
                    loss_weight = [2, 1, 1]

                    outputs=torch.zeros_like(preds[0][:,0,...].unsqueeze(1).float())
                    for i in range(len(preds)):
                        preds_prob = F.softmax(preds[i], dim=1)
                        cur_output = preds_prob[:, 1].unsqueeze(1).float()
                        loss += self.pix_cri(cur_output, labels)# * loss_weight[i]
                        if i==0:
                           outputs=outputs+cur_output
                    #loss /= 3.0
                    #outputs/=3.0

                else:

                    preds_prob = F.softmax(preds, dim=1)
                    outputs = preds_prob[:, 1].unsqueeze(1).float()

                    if self.use_dangle:
                        self.dist_map = sample['dist'].cuda()
                        self.angle_map = sample['angle'].cuda()
                        loss, _ = self.cri_dangle(outputs, pred_mask, pred_dir, labels, self.dist_map,
                                                  self.angle_map)
                    else:
                        #bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                        #loss = bce_loss(outputs, labels)
                        loss=self.pix_cri(outputs,labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc



#=========================for semi-cd trainer===================

class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):

        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
        self.gpu_num = 1#dist.get_world_size()

    def new_global_mask_pooling(self, pred, ignore_mask=None):#[2,19,256,256],[2,256,256]
        return_dict = {}
        n, c, h, w = pred.shape
        # pred_gather = torch.zeros([n * self.gpu_num, c, h, w]).cuda()##[2,19,256,256]
        # #dist.all_gather_into_tensor(pred_gather, pred)
        # pred = pred_gather
        # if ignore_mask is not None:
        #     ignore_mask_gather = torch.zeros([n * self.gpu_num, h, w]).cuda().long()
        #     #dist.all_gather_into_tensor(ignore_mask_gather, ignore_mask)
        #     ignore_mask = ignore_mask_gather
        mask_pred = torch.argmax(pred, dim=1)#[2,256,256]
        pred_softmax = pred.softmax(dim=1)#[2,19,256,256]
        pred_conf = pred_softmax.max(dim=1)[0]#[2,256,256]
        unique_cls = torch.unique(mask_pred)#19
        cls_num = len(unique_cls)#[19]
        new_global = 0.0
        for cls in unique_cls:
            cls_map = (mask_pred == cls)#[2,256,256]
            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:#current cls not found after ignore
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]#[13350]
            cls_max_conf = pred_conf_cls_all.max()#0.2034,
            new_global += cls_max_conf#
        return_dict['new_global'] = new_global / cls_num#0.1361

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global

#=============for class-wise threshold===============

class CB_ThreshController:#use the average prediction of each class to determine threshold, ref:AdaptMatch
    def __init__(self, nclass, momentum, thresh_init=0.95):

        # self.thresh_global0 = torch.tensor(thresh_init).cuda()
        # self.thresh_global1 = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
        self.cls_conf_threshold = [thresh_init for _ in range(nclass)]  # [prob] * 19
        self.gpu_num = 1#dist.get_world_size()
        #self.use_init=use_init

    def init_cb_threshold(self, prob_lu,mask_lu):

        #self.cls_conf_threshold = [prob for _ in range(19)]
        cls_list=torch.arange(self.nclass)
        for cls in cls_list:
            cls_map = (mask_lu == cls)  # [2,256,256]
            if cls_map.sum() == 0:  # current cls not found after ignore
                #cls_num -= 1
                continue
            #cur_threshold = np.percentile(prob_lu.cpu().numpy().flatten(), 80)#20 for entropy
            cur_threshold = np.percentile(prob_lu[cls_map], 80)
            self.cls_conf_threshold[cls]=cur_threshold



    def __call__(self, pred_l, pred_u, mask_mix, prob_mix,use_init=False, ignore_mask=None):#[2,19,256,256],[2,256,256]
        return_dict = {}
        n, c, h, w = pred_l.shape
        pred_l=pred_l.detach()
        pred_u=pred_u.detach()
        prob_l,mask_l=torch.max(F.softmax(pred_l,dim=1),dim=1)
        prob_u, mask_u = torch.max(F.softmax(pred_u, dim=1), dim=1)

        mask_lu=torch.cat([mask_l,mask_u],dim=0)
        prob_lu=torch.cat([prob_l,prob_u],dim=0)

        if use_init:
            self.init_cb_threshold(prob_lu,mask_lu)

        unique_cls = torch.unique(mask_lu)  # 19
        cls_num = len(unique_cls)
        curr_cls_conf_threshold = [0] * 2
        #
        for cls in unique_cls:
            cls_map = (mask_lu == cls)#[2,256,256]
            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:#current cls not found after ignore
                cls_num -= 1
                continue

            pred_conf_cls_all = prob_lu[cls_map]#[13350]
            curr_cls_conf_threshold[cls]=pred_conf_cls_all.mean()
            #update using ema
            self.cls_conf_threshold[cls]= self.momentum * self.cls_conf_threshold[cls] + (1 - self.momentum) * curr_cls_conf_threshold[cls]

        #===========generate mask using cls_threshold================
        mask_thresh=torch.zeros_like(mask_l).cuda()#[4,256,256]
        cls_list=torch.arange(self.nclass)#[0,1]
        for cls in cls_list:
            cur_mask=(mask_mix==cls)#[4,256,256]
            cur_prob_mask=(prob_mix>self.cls_conf_threshold[cls])#[4,2,256,256]
            mask_thresh=mask_thresh+cur_mask*cur_prob_mask
            # cur_mask=prob_mix[mask_mix==cls]
            # mask_thresh[cur_mask>self.cls_conf_threshold[cls]]=1

        return mask_thresh




class TrainerOptimSemi(object):
    # init function for class
    def __init__(self, config,trainDataloader_u, trainDataloader_l, valDataloader,trainDataloader_lu=None
                ):
        dl=RSCD_DL(config)
        self.config=dl.config
        self.model_path=dl.config.model_name
        self.log_file=dl.config.log_path
        self.lossImg_path=dl.config.loss_path

        self.valDataloader = valDataloader
        self.trainDataloader_u = trainDataloader_u
        self.trainDataloader_l = trainDataloader_l
        self.trainDataloader_lu=trainDataloader_lu
        # set the GPU flag
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(0)
        self.mse_loss=nn.MSELoss().cuda(0)
        #criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
        self.criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(0)
        self.criterion_kl = nn.KLDivLoss(reduction='none').cuda(0)
        self.thresh_controller = ThreshController(nclass=config["network_G_CD"]["out_nc"], momentum=0.999, thresh_init=config["train"]["thresh_init"])
        self.cb_thresh_controller=CB_ThreshController(nclass=config["network_G_CD"]["out_nc"], momentum=0.999)



    def load_ck(self,model,model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_dict = checkpoint
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)








    def train_optim_PS_AdaCut_Real(self):#use pseudo loss+adaptive cutmix
        # create model
        start_time = time.perf_counter()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger

        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"iter_epoch":[],
            "loss": [],
                         "acc":[],
                         "val_loss": [],
                         "val_acc": []
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * len(self.trainDataloader_u))
        self.config['train']['niter']=total_iters
        self.config["train"]["lr_steps"] = [int(0.25 * total_iters), int(0.5 * total_iters), int(0.75 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters) - 1]
        model = create_model(self.config)  # create model after updating config
        best_acc = 0
        best_loss=1000
        use_warmup=True if self.config["train"]["warmup_epoch"]>0 else False
        stu_model=model.netG
        tea_model=model.netG_tea

        from data_loaders.transform import transforms_for_noise,transforms_for_rot,transforms_for_scale,transforms_back_rot,transforms_back_scale,postprocess_scale

        #====================for BYOL like contrastive loss======================
        #from losses.feature_memory import FeatureMemory
        #from losses.contrastive_loss import contrastive_class_to_class_learned_memory,NegativeSamplingPixelContrastiveLoss
        from data_loaders.Semiseg_aug import augment_samples
        from .utils import sigmoid_ramp_up,generate_pseudo_box,cut_mix_label_adaptive


        from models.Satt_CD.modules import ramps
        from losses.myLoss import softmax_mse_loss,softmax_mse_loss2,consistency_weight
        rampup_starts = int(self.config["train"]['ramp_up_start'] * self.config['train']['nepoch'])
        rampup_ends = int(self.config["train"]['ramp_up_end'] * self.config['train']['nepoch'])
        iters_per_epoch=len(self.trainDataloader_u)
        cons_w_unsup = consistency_weight(final_w=self.config["train"]['lamda_u'],
                                          iters_per_epoch=iters_per_epoch,
                                          rampup_starts=rampup_starts,
                                          rampup_ends=rampup_ends)  # ramp_val

        cur_epoch=0
        latest_model=self.config.model_dir + '/' + self.config.pred_name + 'latest.pth'
        cur_total_step = cur_epoch*self.config.iter_per_epoch
        #prototype_memory = FeatureMemory(elements_per_class=128, n_classes=2)




        for epoch in range(cur_epoch, total_epochs):
            print('Epoch {}/{}'.format(epoch + 1, total_epochs))
            print('-' * 60)
            epoch_loss = 0
            epoch_acc = 0
            train_history["iter_epoch"].append(epoch)

            if use_warmup or self.config["train"]["lr_scheme"]=="CosineLR" or self.config["train"]["lr_scheme"]=="PolyCycLR" or self.config["train"]["lr_scheme"]=="CosineRe":
                model.update_learning_rate()

            loader = zip(self.trainDataloader_l, self.trainDataloader_u,self.trainDataloader_u)
            cur_total_step=0
            lamda_u=self.config["train"]["lamda_u"]
            '''
            the cumtmix logic for strong aug is:
            let's have an anchor, namely imgA_u_mix, the cutmix of imgA_u_s1 and imgA_u_s2 equals to
            cutmix(imgA_u_s1,imgA_u_mix) and cutmix(imgA_u_s2,imgA_u_mix) in image level
            and 
            cutmix(pred_u_w,pred_u_mix) and cutmix(pred_u_w,pred_u_mix) in label level
            however, the memory consumption rise largely due to image_mix and lalel_mix
            '''
            for current_step, ((imgA_x, imgB_x, mask_x),
                    (imgA_u_w, imgB_u_w, imgA_u_s1, imgB_u_s1,
                     imgA_u_s2, imgB_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                               (imgA_u_w_mix, imgB_u_w_mix, imgA_u_s1_mix,
                                imgB_u_s1_mix, imgA_u_s2_mix, imgB_u_s2_mix, ignore_mask_mix, _, _)
                    ) in enumerate(tqdm(loader, 0)):#enumerate(loader):
                cur_total_step = cur_total_step + 1  # for con_weight
                imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()  # [1,3,256,256]
                imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
                imgA_u_s1, imgB_u_s1 = imgA_u_s1.cuda(), imgB_u_s1.cuda()
                imgA_u_s2, imgB_u_s2 = imgA_u_s2.cuda(), imgB_u_s2.cuda()
                ignore_mask = ignore_mask.cuda()
                cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()  # [1,256,256]
                imgA_u_w_mix, imgB_u_w_mix = imgA_u_w_mix.cuda(), imgB_u_w_mix.cuda()
                imgA_u_s1_mix, imgB_u_s1_mix = imgA_u_s1_mix.cuda(), imgB_u_s1_mix.cuda()
                imgA_u_s2_mix, imgB_u_s2_mix = imgA_u_s2_mix.cuda(), imgB_u_s2_mix.cuda()
                ignore_mask_mix = ignore_mask_mix.cuda()
                #===================generate pseudo label with weak_mix===============================
                #========for rotation consistency====================
                bs = imgA_u_w.shape[1]
                imgs_u = torch.cat([imgA_u_w, imgB_u_w], dim=1)
                imgs_u_noise = transforms_for_noise(imgs_u, 0.5)
                imgs_u_noise, rot_mask, flip_mask = transforms_for_rot(imgs_u_noise)
                imgs_u_noise, scale_mask = transforms_for_scale(imgs_u_noise, self.config["patch_size"])
                with torch.no_grad():  # generate pseudo label with another view of weak unlabel

                    pred_u_w_mix= tea_model(imgA_u_w_mix, imgB_u_w_mix)  # [1,2,256,256]
                    #pred_u_w_mix=F.interpolate(pred_u_w_mix,(256,256),mode='bilinear',align_corners=True)
                    pred_u_w_mix=pred_u_w_mix.detach()
                    conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]  # [1,256,256]
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)  # [1,256,256]
                    #
                    outputs_ema = tea_model(imgs_u_noise[:, :bs, ...], imgs_u_noise[:, bs:, ...])
                    #===================obatin confidence===============
                    # entropy = -torch.sum(pred_u_w_mix.softmax(dim=1) * torch.log(pred_u_w_mix.softmax(dim=1) + 1e-10), dim=1)  # [2,800,800]
                    # entropy /= np.log(2)
                    # confidence = 1.0 - entropy  # [1,800,800]
                    # confidence = confidence * conf_u_w_mix  # [1,800,800]
                    # confidence = confidence.mean(dim=[1, 2])  # 1
                    #confidence = confidence.cpu().numpy().tolist()

                #=============generate pseudo box using pseudo label=========









                # if epoch>=10:#seems not work use such pseudo cutbox========
                #     binary_mask = mask_u_w_mix.data.cpu().numpy() * 255
                #     binary_mask = binary_mask.astype(np.uint8)
                #     pseudo_box1, pseudo_box2 = generate_pseudo_box(binary_mask)
                #     pseudo_box1, pseudo_box2 = pseudo_box1.cuda(), pseudo_box2.cuda()
                #     cutmix_box1,cutmix_box2=pseudo_box1.clone(),pseudo_box2.clone()

                #=====generate labled prediction using weak aug==============
                pred_l= stu_model(imgA_x,imgB_x)
                pred_u_w,pred_u_w_fp,pred_u_w_fp2= stu_model(imgA_u_w, imgB_u_w,need_fp2=True)
                # =============for strong pertubations============



                if np.random.uniform(0,1) < 0.5:
                    binary_mask = mask_x.data.cpu().numpy() * 255
                    binary_mask = binary_mask.astype(np.uint8)
                    pseudo_box1, pseudo_box2 = generate_pseudo_box(binary_mask)
                    pseudo_box1, pseudo_box2 = pseudo_box1.cuda(), pseudo_box2.cuda()
                    cutmix_box1, cutmix_box2 = pseudo_box1.clone(), pseudo_box2.clone()
                    #=============for img cutmix=====================
                    imgA_u_s1[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1] = \
                        imgA_x[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1]
                    imgB_u_s1[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1] = \
                        imgB_x[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1]
                    imgA_u_s2[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1] = \
                        imgA_x[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1]
                    imgB_u_s2[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1] = \
                        imgB_x[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1]

                    #================================pred==========================
                    pred_u_s1 = stu_model(imgA_u_s1, imgB_u_s1)
                    pred_u_s2 = stu_model(imgA_u_s2, imgB_u_s2)

                    conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]  # [1,256,256]
                    mask_u_w = pred_u_w.detach().argmax(dim=1)  # [1,256,256]
                    mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                        mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                    mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                        mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                    #================================================
                    pred_l_mix=pred_l.detach()
                    conf_l_mix = pred_l_mix.softmax(dim=1).max(dim=1)[0]  # [1,256,256]
                    mask_l_mix = pred_l_mix.argmax(dim=1)

                    #==============for label cutmix==================
                    mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_l_mix[cutmix_box1 == 1]
                    conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_l_mix[cutmix_box1 == 1]
                    ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                    mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_l_mix[cutmix_box2 == 1]
                    conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_l_mix[cutmix_box2 == 1]
                    ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]


                else:
                    imgA_u_s1[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1] = \
                        imgA_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgA_u_s1.shape) == 1]
                    imgB_u_s1[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1] = \
                        imgB_u_s1_mix[cutmix_box1.unsqueeze(1).expand(imgB_u_s1.shape) == 1]
                    imgA_u_s2[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1] = \
                        imgA_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgA_u_s2.shape) == 1]
                    imgB_u_s2[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1] = \
                        imgB_u_s2_mix[cutmix_box2.unsqueeze(1).expand(imgB_u_s2.shape) == 1]

                    #=========================pred================================
                    pred_u_s1 = stu_model(imgA_u_s1, imgB_u_s1)
                    pred_u_s2 = stu_model(imgA_u_s2, imgB_u_s2)

                    conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]  # [1,256,256]
                    mask_u_w = pred_u_w.detach().argmax(dim=1)  # [1,256,256]
                    mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                        mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                    mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                        mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                    # =========for cutmix of mask=========
                    mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                    conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
                    ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                    mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                    conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                    ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]



                #====================================================================================


                #==============loss for labeled======================
                loss_l = self.criterion_l(pred_l, mask_x)

                #================loss for unlabeled====================
                # =================for inverse transform===============
                outputs_ema, scale_mask = transforms_back_scale(outputs_ema, scale_mask, self.config["patch_size"])
                #outputs_u = postprocess_scale(pred_u_w, scale_mask, self.config["patch_size"])
                outputs_ema = transforms_back_rot(outputs_ema, rot_mask, flip_mask)
                #======================================================
                outputs_u = F.softmax(pred_u_w, dim=1)
                outputs_ema = F.softmax(outputs_ema, dim=1)
                #===============calculate confidence===================
                # entropy = -torch.sum(outputs_u * torch.log(outputs_u + 1e-10), dim=1)  # [1,800,800]
                # entropy /= np.log(2)
                #================for confidence mask
                # confidence = 1.0 - entropy  # [1,800,800]
                # logits_u_w,_=torch.max(outputs_u,dim=1)
                # confidence = confidence * logits_u_w  # [1,800,800]
                #=======for en_mask===========
                # en_threshold = np.percentile(entropy.detach().cpu().numpy().flatten(), 20)
                # en_mask=(entropy<en_threshold)
                #======================================================
                #consistency_dist=softmax_mse_loss2(outputs_u,outputs_ema,prob_mask=confidence)
                consistency_dist = softmax_mse_loss(outputs_u, outputs_ema)
                loss_u_rot = consistency_dist * cons_w_unsup(cur_total_step)
                #=======================================================
                # pixelWiseWeight = sigmoid_ramp_up(cur_total_step, self.config["train"]["ramp_iter"]) * torch.ones(conf_u_w_cutmixed1.shape).cuda()
                # pixelWiseWeight1 = pixelWiseWeight * torch.pow(conf_u_w_cutmixed1.detach(), 2)
                # pixelWiseWeight2 = pixelWiseWeight * torch.pow(conf_u_w_cutmixed2.detach(), 2)

                loss_u_s1 = self.criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= self.config['train']['conf_thresh']) & (
                            ignore_mask_cutmixed1 != 255))
                #loss_u_s1=loss_u_s1*pixelWiseWeight1
                loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                loss_u_s2 = self.criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= self.config['train']['conf_thresh']) & (
                            ignore_mask_cutmixed2 != 255))
                #loss_u_s2 = loss_u_s2 * pixelWiseWeight2
                loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
                #===========fea loss====================================================
                loss_u_w_fp = self.criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= self.config['train']['conf_thresh']) & (ignore_mask != 255))
                loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

                # loss_u_w_fp2 = self.criterion_u(pred_u_w_fp2, mask_u_w)
                # loss_u_w_fp2 = loss_u_w_fp2 * ((conf_u_w >= self.config['train']['conf_thresh']) & (ignore_mask != 255))
                # loss_u_w_fp2 = loss_u_w_fp2.sum() / (ignore_mask != 255).sum().item()


                mask_valid0 = (mask_x >= 0)
                # loss_en1=entropy_loss(F.softmax(pred_u_s1,dim=1),mask_valid0)*0.01
                # loss_en2 = entropy_loss(F.softmax(pred_u_s2, dim=1), mask_valid0)*0.01
                #=============total loss=========================


                l_g_total = (loss_l + loss_u_rot + (loss_u_s1+ loss_u_s2) * lamda_u*0.25 + (loss_u_w_fp) *lamda_u* 0.5

                            )
                #======================================================================================
                model.optimizer_G.zero_grad()
                l_g_total.backward()
                model.optimizer_G.step()
                if model.optimizer_G_Tea is not None:
                    model.optimizer_G_Tea.step()


                if current_step % self.config['logger']['print_freq'] == 0:
                    # logs = model.get_current_log()
                    epoch_loss += l_g_total.item()

                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}> '.format(
                        epoch, current_step, model.get_current_learning_rate(use_warmup=False),
                        l_g_total.item())

                    logger.info(message)

                #========================for val debug test==========
                # val_loss, val_acc = self.val_CDFormer_Semi(epoch + 1, model=model)
                # message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                # logger.info(message)



            if epoch % self.config['train']['val_epoch'] == 0:

                if self.config["network_G_CD"]["out_nc"] > 1:
                    if self.config["network_G_CD"]["out_nc"] == 2:
                        # val_loss, val_acc = self.val_SEK32(epoch + 1, model=model)
                        val_loss, val_acc = self.val_CDFormer_Semi(epoch + 1, model=model)


                else:
                    val_loss, val_acc = self.val(epoch + 1, model=model)

                message = '<val_loss:{:.6f},val_f1_score:{:.6f}>'.format(val_loss, val_acc)
                logger.info(message)


            train_history["loss"].append(epoch_loss * 1.0 / len(self.trainDataloader_l))
            # train_history["acc"].append(epoch_acc * 1.0 / len(self.trainDataloader))
            train_history["val_loss"].append(val_loss)
            train_history["val_acc"].append(val_acc)
            if self.config["network_G_CD"]["out_nc"] < 10:
                if val_acc > best_acc:
                    best_acc = val_acc
                    model.save_best_acc()
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_best_loss()

        logger.info('Saving final models and training states.')
        model.save(current_step)

        end_time = time.perf_counter()
        run_time=end_time-start_time
        #print(end_time - start_time, 'seconds')
        message='running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim2(train_history)












    def visualize_train_optim(self, history):

        val_acc = history["val_acc"]
        loss = history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        plt.subplot(121)
        #plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        #epoch_array=np.arrange(len(val_acc))
        plt.plot(val_acc)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')


        plt.legend(['train', 'valid'], loc='upper right')
        if self.config.use_KFold:
            lossImg_path=self.config.img_dir + '/' + self.config.pred_name +'_fold'+str(self.config.fold_index)+'.png'
            plt.savefig(lossImg_path)
            plt.close()# close the window before drawing another curve
        else:
           plt.savefig(self.lossImg_path)
           plt.show()

    def visualize_train_optim2(self, history):
        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(hspace=0.3)
        val_acc = history["val_acc"]
        loss = history["loss"]
        acc = history["acc"]
        val_loss = history["val_loss"]

        plt.subplot(121)
        #https://blog.csdn.net/sinat_36219858/article/details/79800460
        # plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        epoch_array=np.arange(len(val_acc))
        plt.plot(epoch_array,val_acc,'r^-')
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['valid'], loc='upper left')
        plt.subplot(122)
        plt.plot(epoch_array,loss,'b*-')
        plt.plot(epoch_array,val_loss,'yo-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.legend(['train', 'valid'], loc='upper right')
        if self.config.use_KFold:
            lossImg_path = self.config.img_dir + '/' + self.config.pred_name + '_fold' + str(
                self.config.fold_index) + '.png'
            plt.savefig(lossImg_path)
            plt.close()  # close the window before drawing another curve
        else:

            plt.savefig(self.lossImg_path)
            plt.show()

            #self.config.model_dir + '/' + self.config.pred_name + '_best_loss.pth'
            save_path=self.config.log_dir + '/' + self.config.pred_name + '_curve'+'.txt'
            from data_loaders.RSCD_dl import MyEncoder
            import json
            f = open(save_path, "w")
            # result = []
            # temp = {}
            # temp['mean'] = data[0]
            # temp['std'] = data[1]
            # result.append(temp)
            f.write(json.dumps(history, cls=MyEncoder))







    def save_checkpoint(self,state, is_best, filename=None):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print("==> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print("==> Validation Accuracy did not improve")





    def  val(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.config['network_G_CD']['which_model_G'] == 'Feat_Cmp':
                    labels0 = F.interpolate(labels, size=torch.Size(
                        [imgs.shape[2] // self.config.ds, imgs.shape[3] // self.config.ds]), mode='nearest')
                    labels0[labels == 1] = -1  # change
                    labels0[labels == 0] = 1  # must convert ot [-1,1] before calculating  loss

                    featT1, featT2 = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    dist = F.pairwise_distance(featT1, featT2, keepdim=True)
                    dist = F.interpolate(dist, size=imgs.shape[2:], mode='bilinear', align_corners=True)

                    outputs = (dist > 1).float()
                    loss = self.cri_dist(dist, labels0)
                elif self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD':
                    _, _, _, outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU' or self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU_DCN' or self.config['network_G_CD']['which_model_G'] == 'EDCls_UNet_BCD_WHU_DCN2' or self.config["network_G_CD"]["which_model_G"] == 'EDCls_UNet_BCD_WHU_Flow' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny' \
                        or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_Tiny_Siam' or self.config["network_G_CD"]["which_model_G"] == 'UNet2D_BCD_CAM_Siam' or self.config["network_G_CD"]["which_model_G"] == 'CD_BIT'\
                        or self.config["network_G_CD"]["which_model_G"] == 'Tiny_CD' or self.config["network_G_CD"]["which_model_G"] == 'CTFINet' or self.config["network_G_CD"]["which_model_G"] == 'SARSNet'\
                        or self.config["network_G_CD"]["which_model_G"] == 'CCLNet':
                    if self.config["network_G_CD"]["use_DS"]:
                       _, _, _, outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])

                    if self.config['network_G_CD']['out_nc']==2:
                        preds_prob = F.softmax(outputs, dim=1)
                        outputs = preds_prob[:, 1].unsqueeze(1).float()

                    bce_loss = bce_edge_loss(use_edge=False).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif self.config['network_G_CD']['which_model_G'] == 'FC_EF' or self.config['network_G_CD']['which_model_G'] == 'Seg_EF' or self.config['network_G_CD']['which_model_G'] == 'UNet_MLP' or self.config['network_G_CD']['which_model_G'] == 'UNetPlusPlus':
                    outputs = val_model(imgs)
                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                elif  self.config['network_G_CD']['which_model_G'] == 'UNet_Trans':
                    outputs = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                    loss = bce_loss(outputs, labels)
                else:
                    _,_,_,outputs=val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    ce_loss=nn.CrossEntropyLoss()
                    loss=ce_loss(outputs,labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc

    def  val_CDFormer(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if self.config["network_G_CD"]["in_c"]==3:
                    if self.use_dangle:
                        preds,pred_mask,pred_dir=val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                    else:
                        preds = val_model(imgs[:, 0:3, ...], imgs[:, 3:6, ...])
                else:
                    preds = val_model(imgs)

                if isinstance(preds,list) or isinstance(preds,tuple):
                    loss = 0
                    loss_weight = [2, 1, 1]

                    outputs=torch.zeros_like(preds[0][:,0,...].unsqueeze(1).float())
                    for i in range(len(preds)):
                        preds_prob = F.softmax(preds[i], dim=1)
                        cur_output = preds_prob[:, 1].unsqueeze(1).float()
                        loss += self.pix_cri(cur_output, labels)# * loss_weight[i]
                        if i==0:
                           outputs=outputs+cur_output
                    #loss /= 3.0
                    #outputs/=3.0

                else:

                    preds_prob = F.softmax(preds, dim=1)
                    outputs = preds_prob[:, 1].unsqueeze(1).float()

                    if self.use_dangle:
                        self.dist_map = sample['dist'].cuda()
                        self.angle_map = sample['angle'].cuda()
                        loss, _ = self.cri_dangle(outputs, pred_mask, pred_dir, labels, self.dist_map,
                                                  self.angle_map)
                    else:
                        #bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                        #loss = bce_loss(outputs, labels)
                        loss=self.pix_cri(outputs,labels)


                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc

    def  val_CDFormer_Semi(self, epoch,model=None,multi_outputs=False):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode
        losses = []
        lossAcc = 0.0
        correctsAcc=0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net
        else:
            val_model=model.netG
            val_model.eval()
        for i, (imgsA,imgsB,labels,_) in (enumerate(tqdm(self.valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():
                # imgs_LR = sample['LR']
                # imgs_HR=sample['HR']
                #imgs,labels=sample['img'],sample['label']
                #imgs, labels = imgs.cuda(), labels.cuda()
                imgsA,imgsB,labels=imgsA.cuda(),imgsB.cuda(),labels.cuda()
                if self.config["network_G_CD"]["in_c"]==3:
                    if self.config["train"]["semi_mode"]=="corrmatch":
                        preds = val_model(imgsA, imgsB)["out"]
                    elif self.config["train"]["semi_mode"]=="TCSM_Con":
                        preds= val_model(imgsA, imgsB)
                        preds=F.interpolate(preds,size=(256,256),mode='bilinear',align_corners=True)
                    else:# or self.config["train"]["semi_mode"]=="PS"
                        preds = val_model(imgsA, imgsB)
                else:
                    preds = val_model(imgsA)

                # if isinstance(preds,list) or isinstance(preds,tuple):
                #     loss = 0
                #     loss_weight = [2, 1, 1]
                #
                #     outputs=torch.zeros_like(preds[0][:,0,...].unsqueeze(1).float())
                #     for i in range(len(preds)):
                #         preds_prob = F.softmax(preds[i], dim=1)
                #         cur_output = preds_prob[:, 1].unsqueeze(1).float()
                #         loss += self.pix_cri(cur_output, labels)# * loss_weight[i]
                #         if i==0:
                #            outputs=outputs+cur_output
                #     #loss /= 3.0
                #     #outputs/=3.0
                #
                # else:
                #
                #     preds_prob = F.softmax(preds, dim=1)
                #     outputs = preds_prob[:, 1].unsqueeze(1).float()
                #
                #     # if self.use_dangle:
                #     #     self.dist_map = sample['dist'].cuda()
                #     #     self.angle_map = sample['angle'].cuda()
                #     #     loss, _ = self.cri_dangle(outputs, pred_mask, pred_dir, labels, self.dist_map,
                #     #                               self.angle_map)
                #     # else:
                #     #     #bce_loss = bce_edge_loss(use_edge=True).to(self.device)
                #     #     #loss = bce_loss(outputs, labels)

                loss=self.criterion_l(preds,labels)

                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                preds_prob = F.softmax(preds, dim=1)
                outputs = preds_prob[:, 1].float()
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc+=f1_score

        val_loss=lossAcc*1.0/(len(self.valDataloader))
        val_acc=correctsAcc*1.0/(len(self.valDataloader))

        print('Epoch %d evaluate done ' % epoch)
        # convert to train mode for next training
        if model==None:
            self.net.train()

        del outputs
        torch.cuda.empty_cache()

        return  val_loss,val_acc