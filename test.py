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
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

from models import *

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='data directory', default='./data')
    parser.add_argument('--data-suffix', type=int, help='data version', default=3)
    parser.add_argument('--model', type=str, help="Model: ['lgb', 'kernelridge', 'gradientboosting', 'knn']",
                        choices=['lgb', 'kernel_ridge', 'gradient_boosting', 'knn'], default='lgb')
    parser.add_argument('--model-dir', help='model directory', default='./model')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    print(args.data_dir)

    ###############################
    ########## Load Data ##########
    ###############################

    train_x = pd.read_pickle(os.path.join(args.data_dir, 'train_x_v2.pkl'))
    train_y = pd.read_pickle(os.path.join(args.data_dir, 'train_y_v2.pkl'))
    val_x = pd.read_pickle(os.path.join(args.data_dir, 'val_x_v2.pkl'))
    val_y = pd.read_pickle(os.path.join(args.data_dir, 'val_y_v2.pkl'))
    test_x = pd.read_pickle(os.path.join(args.data_dir, 'test_x_v2.pkl'))
    test_y = pd.read_pickle(os.path.join(args.data_dir, 'test_y_v2.pkl'))
    print("Input Features: ", test_x.columns)
    print("Prediction Features: ", test_y.columns)

    if args.model == 'lgb':
        model = LGBTrainer(boosting_type='gbdt', metric='rmse', lr=0.1, epoch=100)
        test_y_pred = model.test(args.model_dir, test_x)
    elif args.model == 'kernel_ridge':
        model = KRTrainer(kernel='rbf', alpha=1.0)
        test_y_pred = model.test(args.model_dir, train_x, train_y, test_x)



    print("Prediction: \n", test_y_pred)
    print("\nTrue: \n", test_y)