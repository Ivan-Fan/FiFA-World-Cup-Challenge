from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

from functools import partial
import glob
import sklearn
import sys
import joblib

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier

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
# from sklearn.linear_model import PoissonRegressor
from utils import evaluate

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
            X_train = train_x

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

            rmse += scores['rmse']
            rmsle += scores['rmsle']
            r2 += scores['r2']

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


    def test(self, model_dir, train_x, train_y, test_x):

        y_preds = pd.DataFrame()

        for target in ['home_score', 'away_score']:
            model_path = os.path.join(model_dir, 'lgb_model_' + target + '.txt')
            clf = lgb.Booster(model_file=model_path)
            y_preds[target] = clf.predict(test_x)
            y_preds[target][y_preds[target] < 0] = 0

        return y_preds


class LGBClassifierTrainer:

    def __init__(self, num_classes, boosting_type='gbdt', metric='multiclass', lr=0.1, epoch=100):

        self.boosting_type = boosting_type
        self.metric = metric
        self.lr = lr
        self.epoch = epoch
        self.num_classes = num_classes

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
                'objective': self.metric, # multiclass
                'metric': 'multi_logloss',
                # 'max_depth': 10,
                'learning_rate': self.lr,
                'verbose': 0,
                'num_threads': 8,
                'num_classes': self.num_classes,
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
            X_train = train_x

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(val_x, label=y_val)

            clf = lgb.train(params, dtrain, self.epoch, early_stopping_rounds=10, valid_sets=[dtrain, dvalid])

            feature_importances = pd.DataFrame()
            feature_importances['feature'] = train_x.columns

            feature_importances['importance'] = clf.feature_importance()

            y_pred_valid = np.argmax(np.array(clf.predict(val_x)), axis=1)
            y_pred_valid[y_pred_valid < 0] = median_target

            scores['rmse'] = mean_squared_error(y_val, y_pred_valid, squared=False)
            scores['rmsle'] = mean_squared_log_error(y_val, y_pred_valid, squared=False)
            scores['r2'] = r2_score(y_val, y_pred_valid)
            print(f"RMSE: {scores['rmse']} | RMSLE: {scores['rmsle']} | R2: {scores['r2']}")

            rmse += scores['rmse']
            rmsle += scores['rmsle']
            r2 += scores['r2']

            clf.save_model(os.path.join(save_dir, 'lgb_classifier_' + target + '.txt'))

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


    def test(self, model_dir, train_x, train_y, test_x):

        y_preds = pd.DataFrame()

        for target in ['home_score', 'away_score']:
            model_path = os.path.join(model_dir, 'lgb_classifier_' + target + '.txt')
            clf = lgb.Booster(model_file=model_path)
            y_preds[target] = np.argmax(clf.predict(test_x),axis=1)
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
        clf = MultiOutputRegressor(KernelRidge(kernel=self.kernel, alpha=self.alpha))
        clf.fit(train_x, train_y)
        joblib.dump(clf, os.path.join(save_dir, "KR_model.pkl"))


    def test(self, model_dir, train_x, train_y, test_x):
        clf = joblib.load(os.path.join(model_dir, "KR_model.pkl"))
        y_preds = clf.predict(test_x)
        y_preds[y_preds < 0] = 0

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


class PoissonRegressor:
    def __init__(self, alpha: 0.1):
        self.alpha = alpha

    def train(self, train_x, train_y, val_x, val_y, save_dir):

        # uncomment for grid search
        # clf = MultiOutputRegressor(PoissonRegressor())
        # alpha = [0.1, 0.2]
        # params = {'estimator__alpha': alpha}
        # tscv = TimeSeriesSplit(n_splits=5)
        # regressor = GridSearchCV(clf, params, cv=tscv, refit=False)
        # regressor.fit(train_x, train_y)
        # print("Best parameters: {}".format(regressor.best_params_))
        # print("Results: {}".format(regressor.cv_results_))

        clf = MultiOutputRegressor(PoissonRegressor(alpha=self.alpha))
        clf.fit(train_x, train_y)
        # save
        joblib.dump(clf, os.path.join(save_dir, "Poisson_model.pkl"))

    def test(self, model_dir, train_x, train_y, test_x):
        clf = joblib.load(os.path.join(model_dir, "Poisson_model.pkl"))
        y_preds = clf.predict(test_x)
        return y_preds


class GradientBoostTrainer:
    def __init__(self, lr=0.05, n_estimators=100):
        self.lr = lr
        self.n_estimators = n_estimators

    def train(self, train_x, train_y, val_x, val_y, save_dir):
        # gbr = MultiOutputRegressor(GradientBoostingRegressor())
        #
        # lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
        # params = {'estimator__learning_rate': lrs, 'estimator__n_estimators': [100]}
        # tscv = TimeSeriesSplit(n_splits=5)
        # regressor = GridSearchCV(gbr, params, cv=tscv, refit=False)
        # print("Best parameters: {}".format(regressor.best_params_))
        # print("Results: {}".format(regressor.cv_results_))

        gbr = MultiOutputRegressor(GradientBoostingRegressor(
            learning_rate=self.lr, n_estimators=self.n_estimators))
        gbr.fit(train_x, train_y)
        joblib.dump(gbr, os.path.join(save_dir, "GradientBoost_model.pkl"))

    def test(self, model_dir, train_x, train_y, test_x):
        gbr = joblib.load(os.path.join(model_dir, "GradientBoost_model.pkl"))
        y_preds = gbr.predict(test_x)
        return y_preds


class RandomForestTrainer:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def train(self, train_x, train_y, val_x, val_y, save_dir):
        # rfr = MultiOutputRegressor(RandomForestRegressor())
        #
        # dpt = [2, 6, 10]
        # params = {"estimator__max_depth": dpt,
        #           # "estimator__min_samples_split": [2, 3, 10],
        #           # "estimator__min_samples_leaf": [1, 3, 10],
        #           "estimator__bootstrap": [True, False],
        #           "estimator__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]}
        # tscv = TimeSeriesSplit(n_splits=5)
        # regressor = GridSearchCV(rfr, params, cv=tscv, refit=False)
        # regressor.fit(train_x, train_y)
        # print("Best parameters: {}".format(regressor.best_params_))
        # print("Results: {}".format(regressor.cv_results_))

        rfr = MultiOutputRegressor(RandomForestRegressor(
            max_depth=2))
        rfr.fit(train_x, train_y)
        joblib.dump(rfr, os.path.join(save_dir, "RandomForest_model.pkl"))

    def test(self, model_dir, train_x, train_y, test_x):
        rfr = joblib.load(os.path.join(model_dir, 'RandomForest_model.pkl'))
        y_preds = rfr.predict(test_x)
        return y_preds

class RandomForestClassifierTrainer:
    def __init__(self, max_depth=6):
        self.max_depth = max_depth

    def train(self, train_x, train_y, val_x, val_y, save_dir):
        # rfc = MultiOutputClassifier(RandomForestClassifier())
        #
        # dpt = [2, 6, 10]
        # params = {"estimator__max_depth": dpt,
        #           # "estimator__min_samples_split": [2, 3, 10],
        #           # "estimator__min_samples_leaf": [1, 3, 10],
        #           "estimator__criterion": ["gini", "entropy", "log_loss"]}
        # tscv = TimeSeriesSplit(n_splits=5)
        # classifier = GridSearchCV(rfc, params, cv=tscv, refit=False)
        # classifier.fit(train_x, train_y)
        # print("Best parameters: {}".format(classifier.best_params_))
        # print("Results: {}".format(classifier.cv_results_))

        rfr = MultiOutputClassifier(RandomForestClassifier(
            max_depth=self.max_depth, criterion="entropy"))
        rfr.fit(train_x, train_y)
        joblib.dump(rfr, os.path.join(save_dir, "RandomForest_classifier_model.pkl"))

    def test(self, model_dir, train_x, train_y, test_x):
        rfr = joblib.load(os.path.join(model_dir, 'RandomForest_classifier_model.pkl'))
        y_preds = rfr.predict(test_x)
        return y_preds

from statsmodels.tsa.arima.model import ARIMA
class ARIMATrainer:
    def __init__(self):
        self.data_dir = 'data/V3'
        self.time_series_data = dict()

    def load_time_series(self):
        for team in ['Argentina', 'Netherlands', 'Croatia', 'Brazil', 'Morocco', 'Portugal', 'England', 'France']:
            self.time_series_data[team] = pd.read_pickle(os.path.join(self.data_dir, '{}.pkl'.format(team)))

    def train_test(self, save_dir):
        self.load_time_series()
        # Stationary Analysis
        fig = plt.figure()
        plt.plot(self.time_series_data['Argentina'])
        plt.xlabel("Time Step")
        plt.ylabel("Scores")
        plt.title("History Performance of Argentina")

        fig = plt.figure()
        plt.plot(self.time_series_data['Netherlands'])
        plt.xlabel("Time Step")
        plt.ylabel("Scores")
        plt.title("History Performance of Netherlands")

        arima_models, arima_predictions = dict(), dict()
        mses, rmsles, r2s = dict(), dict(), dict()
        for k, v in self.time_series_data.items():
            model = ARIMA(self.time_series_data[k], order=(1, 1, 2))
            trained_model = model.fit()
            arima_models[k] = trained_model

            train_score = self.time_series_data[k]
            train_predicted_score = trained_model.predict(start=0, end=len(self.time_series_data[k]) - 1)
            mses[k], rmsles[k], r2s[k] = evaluate(train_score, train_predicted_score)
            predicted_score = trained_model.predict(len(self.time_series_data[k]))
            arima_predictions[k] = float(predicted_score)

        print(f"Train set MSE = ")
        print(mses)
        print(f"Train set RMSLE = ")
        print(rmsles)
        print(f"Train set R2 = ")
        print(r2s)
        print("Prediction =")
        print(arima_predictions)

from hmmlearn import hmm
import pickle
import collections
class HmmTrainer:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.data_dir = 'data/V3'
        self.time_series_data = dict()

    def train_test(self, save_dir):
        scores = collections.defaultdict(list)
        mses, rmsles, r2s = {}, {}, {}
        for team in ['Argentina', 'Netherlands', 'Croatia', 'Brazil', 'Morocco', 'Portugal', 'England', 'France']:
            for n_components in range(1, self.num_classes):
                print("hmm_" + str(team) + ".pkl")
                remodel = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=20)
                self.time_series_data[team] = pd.read_pickle(os.path.join(self.data_dir, '{}.pkl'.format(team)))
                remodel.fit(self.time_series_data[team])
                curr = remodel.score(self.time_series_data[team])
                if not scores[team] or max(scores[team]) < curr:
                    with open(os.path.join(save_dir, "hmm_" + str(team) + ".pkl"), "wb") as file:
                        pickle.dump(remodel, file)
                scores[team].append(curr)
                print(f'Converged: {remodel.monitor_.converged}\t\t'
                      f'Score: {scores[team][-1]}')

            states = remodel.predict(self.time_series_data[team])
            # mses[team], rmsles[team], r2s[team] = evaluate(self.time_series_data[team], remodel.lambdas_[states])

        preds = {}
        for team in ['Argentina', 'Netherlands', 'Croatia', 'Brazil', 'Morocco', 'Portugal', 'England', 'France']:
            with open(os.path.join("models", "hmm_" + str(team) + ".pkl"), "rb") as file:
                remodel = pickle.load(file)
                preds[team] = remodel.sample(1)[0]

        # print(f"Train set MSE = ")
        # print(mses)
        # print(f"Train set RMSLE = ")
        # print(rmsles)
        # print(f"Train set R2 = ")
        # print(r2s)
        print("Prediction =")
        print(preds)


