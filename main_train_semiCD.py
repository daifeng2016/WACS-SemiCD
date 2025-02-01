from configs.config_utils import process_config, get_train_args
import numpy as np

from trainers.trainer_optim_CD import TrainerOptimSemi
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
from sklearn.model_selection import train_test_split
from data_loaders.data_proc import SemiCDDataset,CB_SemiCDDataset
import cv2
seed=37148
rng = np.random.RandomState(seed)#
import argparse
from configs.config_utils import get_config_from_json
import utils.logger as Logger
import logging
logger = logging.getLogger('base')
from sklearn.model_selection import KFold
# torch.manual_seed(seed)##为CPU设置随机种子
# #torch.cuda.manual_seed()#为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)##设置用于在所有GPU上生成随机数的种子。 如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。为所有GPU设置随机种子can make the result the same, but the acc is lower

def parse_option(cur_root,up_root):
    parser_file = argparse.ArgumentParser('argument for training')
    # specify folder
    parser_file.add_argument('--input_path', type=str, default=up_root+'/val', help='path to input data')
    parser_file.add_argument('--output_path', type=str, default=up_root+'/output', help='path to output data')
    parser_file.add_argument('--config_path', type=str, default=cur_root+'/configs/config.json', help='path to json')

    args_file = parser_file.parse_args()

    config, _ = get_config_from_json(args_file.config_path)
    return args_file,config



def main_train():

    parser = None
    config = None

    cur_root = os.path.dirname(os.path.abspath(__file__))
    up_root = os.path.abspath(os.path.join(cur_root, ".."))
    args_file, config = parse_option(cur_root, up_root)  # get input and outputfile


    #===========================================
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True#使用benchmark以启动CUDNN_FIND自动寻找最快的操作，当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能


    unlabeld_path = os.path.join(config.data_dir, 'splits', config["train"]["semi_ratio"], 'unlabeled.txt')
    labeld_path = os.path.join(config.data_dir, 'splits', config["train"]["semi_ratio"], 'labeled.txt')
    if "CB" in config["train"]["semi_mode"]:
        trainset_u = CB_SemiCDDataset(config.data_dir, 'train_u',
                                      config.patch_size, unlabeld_path, dataset=config.dataset_name)
        trainset_l = CB_SemiCDDataset(config.data_dir, 'train_l',
                                      config.patch_size, labeld_path, nsample=len(trainset_u.ids),
                                      dataset=config.dataset_name)
    else:
        trainset_u = SemiCDDataset(config.data_dir, 'train_u',
                                   config.patch_size, unlabeld_path, dataset=config.dataset_name)
        trainset_l = SemiCDDataset(config.data_dir, 'train_l',
                                   config.patch_size, labeld_path, nsample=len(trainset_u.ids),
                                   dataset=config.dataset_name)
    #=============================================================
    valset = SemiCDDataset(config.data_dir, 'val', dataset=config.dataset_name)



    trainloader_l = DataLoader(trainset_l, batch_size=config.batch_size,
                               pin_memory=True, num_workers=1, drop_last=True  # , sampler=trainsampler_l
                               )
    trainloader_u = DataLoader(trainset_u, batch_size=config.batch_size,
                               pin_memory=True, num_workers=1, drop_last=True  # , sampler=trainsampler_u
                               )
    valloader = DataLoader(valset, batch_size=config.batch_size*4, pin_memory=True, num_workers=1,
                           drop_last=False
                           )

    print("the number of trainset-unlabeled is %d" % len(trainset_u.ids))
    print("the number of trainset-labeled is %d" % trainset_l.set_num)
    print("the number of valSet is %d" % len(valset.ids))
    logger.info("the number of trainset-unlabeled is {}".format(len(trainset_u.ids)))
    logger.info("the number of trainset-labeled is {}".format((trainset_l.set_num)))
    logger.info("the number of valSet is {}".format(len(valset.ids)))


    #==========================for traning using optim of each iteration==========================
    trainer = TrainerOptimSemi(config, trainloader_u, trainloader_l, valloader)
    trainer.train_optim_PS_AdaCut_Real()


if __name__ == '__main__':
    main_train()