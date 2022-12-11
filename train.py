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
from utils import evaluate

from models import *

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', help='data directory', default='data/V2')
    parser.add_argument('--data-suffix', type=int, help='data version', default=3)
    parser.add_argument('--model', type=str, help="Model: ['lgb', 'lgb_classifier', 'kernel_ridge', 'gradient_boost', 'knn', 'poisson', 'random_forest', 'random_forest_classifier']",
                        choices=['lgb','lgb_classifier', 'kernel_ridge', 'gradient_boost', 'knn', 'poisson', 'random_forest', 'random_forest_classifier'], default='lgb')
    parser.add_argument('--cur-year', type=int, help='test data version', default=2018)
    parser.add_argument('--model-dir', help='Directory for saving trained model files', default='models')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    ###############################
    ########## Load Data ##########
    ###############################
    train_x = pd.read_pickle(os.path.join(args.data_dir, 'train_x.pkl'))
    train_y = pd.read_pickle(os.path.join(args.data_dir, 'train_y.pkl'))
    val_x = pd.read_pickle(os.path.join(args.data_dir, 'val_x.pkl'))
    val_y = pd.read_pickle(os.path.join(args.data_dir, 'val_y.pkl'))
    test_x = pd.read_pickle(os.path.join(args.data_dir, 'test_{}_x.pkl'.format(args.cur_year)))
    test_y = pd.read_pickle(os.path.join(args.data_dir, 'test_{}_y.pkl'.format(args.cur_year)))
    print("Input Features: ", train_x.columns)
    print("Prediction Features: ", train_y.columns)

    # print(np.max(np.concatenate([train_y, val_y, test_y], axis = 0)))
    num_classes = int(np.max(np.concatenate([train_y, val_y, test_y], axis = 0))) + 1

    if args.model == 'lgb':
        model = LGBTrainer(boosting_type='gbdt', metric='rmse', lr=0.1, epoch=100)
    elif args.model == 'lgb_classifier':
        model = LGBClassifierTrainer(num_classes=num_classes, boosting_type='gbdt', metric='multiclass', lr=0.1, epoch=100)
    elif args.model == 'kernel_ridge':
        model = KRTrainer(kernel='rbf', alpha=0.2)
    elif args.model == 'knn':
        model = KNNTrainer(n_neighbors=[3, 5, 7, 9])
    elif args.model == 'poisson':
        model = PoissonRegressor(alpha=0.1)
    elif args.model == 'gradient_boost':
        model = GradientBoostTrainer(lr=0.05, n_estimators=100)
    elif args.model == 'random_forest':
        model = RandomForestTrainer(max_depth=2)
    elif args.model == 'random_forest_classifier':
        model = RandomForestClassifierTrainer(max_depth=6)


    ###############################
    ##### Train and Evaluate ######
    ###############################

    model.train(train_x, train_y, val_x, val_y, args.model_dir)

    # show the train metric / val metric
    train_mse, train_rmsle, train_r2 = evaluate(train_y, model.test(args.model_dir, train_x, train_y, train_x))
    val_mse, val_rmsle, val_r2 = evaluate(val_y, model.test(args.model_dir, train_x, train_y, val_x))
    test_mse, test_rmsle, test_r2 = evaluate(test_y, model.test(args.model_dir, train_x, train_y, test_x))
    print("test: \n",test_y)
    print("pred: \n", model.test(args.model_dir, train_x, train_y, test_x))
    print(f"Train set MSE = {train_mse}")
    print(f"Train set RMSLE = {train_rmsle}")
    print(f"Train set R2 = {train_r2}")
    print(f"Valid set MSE = {val_mse}")
    print(f"Valid set RMSLE = {val_rmsle}")
    print(f"Valid set R2 = {val_r2}")
    print(f"Test set MSE = {test_mse}")
    print(f"Test set RMSLE = {test_rmsle}")
    print(f"Test set R2 = {test_r2}")
