from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

from functools import partial
import glob
import sklearn
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import random, datetime
from sklearn import metrics
import gc
import lightgbm as lgb
import sklearn

import argparse


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='Model version name', default='ASRf0176_test')
    parser.add_argument('--model', type=str, help="Model structure: ['ASRLight', 'ASR']",
                        choices=['ASRLight', 'ASR'], default='ASRLight')
    parser.add_argument('--exp-type', type=str,
                        help="Experiment type: ['WordErrSeq', 'WordErr', 'ASR', 'Test', 'ASRf0176', 'ASRf0176_2', 'ASRhw']",
                        choices=['WordErrSeq', 'WordErr', 'ASR', 'Test', 'ASRf0176', 'ASRf0176_2', 'ASRhw'], default='ASRf0176')
    parser.add_argument('--train-type', type=str, help="Train type",
                        choices=['train', 'eval', 'debug'], default='train')
    parser.add_argument('--data-path', help="Data path: './data/recognition/f0176' or './data/recognition/'", 
                        default="./data/recognition/f0176/") 
    parser.add_argument('--data-suffix', type=str, help="half 1 or half 2")
    parser.add_argument('--result-path', type=str, help='Result directory', default='./result')
    parser.add_argument('--log-path', type=str, help='Log directory', default='./log')
    parser.add_argument('--exp-path', type=str, help='Log directory', default='./exp')
    parser.add_argument('--run-path', type=str, help='Tensorboard directory', default='./runs')
    parser.add_argument('--ckpt-path', type=str, help='Checkpoint directory', default='./checkpoint')

    return parser



if __name__ == '__main__':

    args = parse_args()
    
    train_x = pd.read_pickle(os.path.join(save_dir, 'train_x_v2.pkl'))
    train_y = pd.read_pickle(os.path.join(save_dir, 'train_y_v2.pkl'))
    val_x = pd.read_pickle(os.path.join(save_dir, 'val_x_v2.pkl'))
    val_y = pd.read_pickle(os.path.join(save_dir, 'val_y_v2.pkl'))
    test_x = pd.read_pickle(os.path.join(save_dir, 'test_x_v2.pkl'))
    test_y = pd.read_pickle(os.path.join(save_dir, 'test_y_v2.pkl'))
    print(train_x.columns, train_y.columns)