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
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from typing import List

class LGBTrainer:

    def __init__(self, boosting_type='gbdt', metric='rmse', lr=0.1, epoch=100):

        self.boosting_type = boosting_type
        self.metric = metric
        self.lr = lr
        self.epoch = epoch

    def train(self, train_x, train_y, val_x, val_y, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        input_feat = train_x.columns
        output_feat = train_y.columns

        print("Start training!")
        for target in ['home_score', 'away_score']:
            y_train = train_y[target]
            y_val = val_y[target]

            # y_oof = np.zeros(train_x.shape[0])
            rmse = 0
            rmsle = 0
            r2 = 0

            params = {
                'boosting_type': self.boosting_type,
                'objective': self.metric,
                'metric': 'rmse',
                'max_depth': 6,
                'learning_rate': self.lr,
                'verbose': 0,
                'num_threads': 8,
                'n_estimators': 200}

            # model = lgb.LGBMRegressor()

            median_target = np.median(train_y.values)
            feature_importances = pd.DataFrame()
            feature_importances['feature'] = input_feat

            scores = {
                'rmse': 0.0,
                'rmsle': 0.0,
                'r2': 0.0,
            }
            X_train = train_x  # [columns]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(val_x, label=y_val)

            clf = lgb.train(params, dtrain, self.epoch, early_stopping_rounds=50, valid_sets=[dtrain, dvalid])

            feature_importances = pd.DataFrame()
            feature_importances['feature'] = train_x.columns

            feature_importances['importance'] = clf.feature_importance()

            y_pred_valid = clf.predict(val_x)
            y_pred_valid[y_pred_valid < 0] = median_target

            scores['rmse'] = mean_squared_error(y_val, y_pred_valid, squared=False)
            scores['rmsle'] = mean_squared_log_error(y_val, y_pred_valid, squared=False)
            scores['r2'] = r2_score(y_val, y_pred_valid)
            print(f"RMSE: {scores['rmse']} | RMSLE: {scores['rmsle']} | R2: {scores['r2']}")

            rmse += scores['rmse']  # / NFOLDS
            rmsle += scores['rmsle']  # / NFOLDS
            r2 += scores['r2']  # / NFOLDS

            clf.save_model(os.path.join(save_dir, 'lgb_model_' + target + '.txt'))

            del X_train, y_train
            gc.collect()

            ###############################
            ##### Feature Analysis ########
            ###############################

        sns.set()
        plt.figure(figsize=(20, 5))
        sns.barplot(data=feature_importances.sort_values(by='importance', ascending=False).head(10), x='importance',
                    y='feature')
        plt.title('TOP feature importance')
        plt.show()


    def test(self, model_dir, test_x):

        y_preds = pd.DataFrame()

        for target in ['home_score', 'away_score']:
            model_path = os.path.join(model_dir, 'lgb_model_' + target + '.txt')
            clf = lgb.Booster(model_file=model_path)
            y_preds[target] = clf.predict(test_x)
            y_preds[target][y_preds[target] < 0] = 0

        return y_preds


class KRTrainer:

    def __init__(self, kernel='rbf', alpha=0.2):

        self.kernel = kernel
        self.alpha = alpha

    def train(self, train_x, train_y, val_x, val_y, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        input_feat = train_x.columns
        output_feat = train_y.columns

        print("Start training!")
        for target in ['home_score', 'away_score']:

            scores = {
                'rmse': 0.0,
                'rmsle': 0.0,
                'r2': 0.0,
            }
            clf = KernelRidge(kernel=self.kernel, alpha=self.alpha)
            clf.fit(train_x, train_y[target])

            y_pred_valid = clf.predict(val_x)
            y_pred_valid[y_pred_valid < 0] = 0
            y_val = val_y[target]

            scores['rmse'] = mean_squared_error(y_val, y_pred_valid, squared=False)
            scores['rmsle'] = mean_squared_log_error(y_val, y_pred_valid, squared=False)
            scores['r2'] = r2_score(y_val, y_pred_valid)
            print(f"{target} | RMSE: {scores['rmse']} | RMSLE: {scores['rmsle']} | R2: {scores['r2']}")

            clf_param = clf.get_params()
            print(clf_param)
            np.save(os.path.join(save_dir, 'KR_model_' + target + '.npy'), clf_param)

    def test(self, model_dir, train_x, train_y, test_x):

        y_preds = pd.DataFrame()

        for target in ['home_score', 'away_score']:

            clf = KernelRidge(kernel=self.kernel, alpha=self.alpha)
            # clf.fit(train_x, train_y[target])
            model_path = os.path.join(model_dir, 'KR_model_' + target + '.npy')
            clf_param = np.load(model_path, allow_pickle=True)
            print(clf_param.dtype)
            clf.set_params(**clf_param)
            y_preds[target] = clf.predict(test_x)
            y_preds[target][y_preds[target] < 0] = 0

        return y_preds

class KNNTrainer:

    def __init__(self, n_neighbors: List):

        self.n_neighbors = n_neighbors if n_neighbors else [3, 5, 7, 9]

    def train(self, train_x, train_y, val_x, val_y, save_dir):

        os.makedirs(save_dir, exist_ok=True)
        input_feat = train_x.columns
        output_feat = train_y.columns

        knn = MultiOutputClassifier(KNeighborsClassifier())

        n_neighbors = self.n_neighbors
        weights = ['uniform', 'distance']
        params = {'estimator__n_neighbors': n_neighbors,
                  'estimator__weights': weights}
        clf = GridSearchCV(knn, params, refit=True)
        clf.fit(train_x, train_y)

        print("Best parameters: {}".format(clf.best_params_))

        val_y_preds = clf.predict(val_x)

        print(f"Val set RMSE = {mean_squared_error(val_y, val_y_preds, squared=False)}")
        print(f"Val set RMSLE = {mean_squared_log_error(val_y, val_y_preds, squared=False)}")
        print(f"Val set R2 = {r2_score(val_y, val_y_preds)}")

    def test(self, model_dir, train_x, train_y, test_x):

        y_preds = pd.DataFrame()

        for target in ['home_score', 'away_score']:
            # model_path = os.path.join(model_dir, 'KR_model_' + target + '.npy')
            # clf_param = np.load(model_path, allow_pickle=True)

            clf = KernelRidge(kernel=self.kernel, alpha=self.alpha)
            clf.fit(train_x, train_y[target])

            y_preds[target] = clf.predict(test_x)
            y_preds[target][y_preds[target] < 0] = 0

        return y_preds