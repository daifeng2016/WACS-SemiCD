# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by D. F. Peng on 2019/6/18
"""
import argparse
import json
from collections import OrderedDict

import os
from bunch import Bunch

from configs.root_dir import ROOT_DIR
from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    # remove comments starting with '//'
    json_str = ''
    with open(json_file, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    config_dict = json.loads(json_str, object_pairs_hook=OrderedDict)

    # with open(json_file, 'r') as config_file:
    #     config_dict = json.load(config_file)  # 配置字典
    #


    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(json_file):
    """
    解析Json文件
    :param json_file: 配置文件
    :return: 配置类
    """
    config, _ = get_config_from_json(json_file)

    #exp_dir = os.path.join(ROOT_DIR, "experiments")# join 函数连接不需要添加\
    #mkdir_if_not_exist(exp_dir)  # 创建文件夹
    #config 为bunch对象，可以自动增加对象属性
    # config.tb_dir = os.path.join(exp_dir, config.exp_name, "logs/")  # 日志
    # config.cp_dir = os.path.join(exp_dir, config.exp_name, "checkpoints/")  # 模型
    # config.img_dir = os.path.join(exp_dir, config.exp_name, "images/")  # 存储网络图，结果图
    # config.data_dir = os.path.join(exp_dir, config.exp_name, "data/")  # 存储训练和测试样本 .npy格式

    # mkdir_if_not_exist(config.tb_dir)  # 创建文件夹
    # mkdir_if_not_exist(config.cp_dir)  # 创建文件夹
    # mkdir_if_not_exist(config.img_dir)  # 创建文件夹
    # mkdir_if_not_exist(config.data_dir)
    return config


def get_train_args():
    """
    添加训练参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',#文件属性 path extension exclude
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():
    """
    添加测试路径
    :return: 参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='C',
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser
