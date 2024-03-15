# Scripts for tuning hyperparameters for Random Forest.

import pandas as pd
import os
import numpy as np
import random
from boruta import BorutaPy
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
from sklearn import model_selection
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import optuna
from data_preprocessing import read_csv, seperate_x_y, split_train_test
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
import sklearn.neighbors._base
import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from imblearn.over_sampling import BorderlineSMOTE
import model_setting

import warnings

warnings.filterwarnings("ignore")


def base_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, default='0',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=0)
    parser.add_argument('--file_path', type=str,
                        default="../Data/Processed/Merged_pateint_file_25p.csv")
    parser.add_argument('--save_path', type=str,
                        default='../models/Random_Forest/')
    parser.add_argument('--exp_name', type=str,
                        default='Random Forest model')
    parser.add_argument('--PFS_threshold', type=float,
                        default=7.1)
    parser.add_argument('--split', type=float,
                        default=0.3)
    parser.add_argument('--CV', type=int,
                        default=5)
    parser.add_argument('--mode', type=str,
                        default='train', choices=['tune_hyperparas', 'train', 'test'])
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--n_estimators', type=int,
                        default=300)
    parser.add_argument('--max_depth', type=int,
                        default=75)
    parser.add_argument('--min_samples_split', type=int,
                        default=2)
    parser.add_argument('--min_samples_leaf', type=int,
                        default=1)
    parser.add_argument('--max_features', type=str,
                        default='sqrt')
    parser.add_argument('--number_trial', type=int,
                        default=100)
    parser.add_argument('--save_freq', type=int,
                        default=5)
    config = parser.parse_args()
    return config


def load_data(config):
    df = read_csv(config.file_path)
    x, y = seperate_x_y(df, config.PFS_threshold)
    X_train, X_test, y_train, y_test = split_train_test(x, y)
    X_train = read_csv("../Data/Processed/imputed_train_25p.csv")  # imp_fs_vif_train_features_20p
    X_test = read_csv("../Data/Processed/imputed_test_25p.csv")  # imp_fs_vif_test_features_20p
    y_train = pd.DataFrame(y_train)
    X_train, y_train = smote(X_train, y_train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def smote(X_train, y_train, random_state: int = 0):
    smote_border = SMOTE(random_state=random_state)
    X_smoteborder, y_smoteborder = smote_border.fit_resample(X_train, y_train)
    X_smote_df = pd.DataFrame(X_smoteborder, columns=X_train.columns)
    y_smote_df = pd.DataFrame(y_smoteborder, columns=y_train.columns)
    return X_smote_df, y_smoteborder


def objective(trial, config):
    """ objective function"""

    # main hyperparameters
    config.n_estimators = trial.suggest_categorical('n_estimators', [5, 10, 20, 30, 50, 60])
    config.max_depth = trial.suggest_int('max_depth', 3, 20, step=5)
    config.min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    config.min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4])
    config.max_features = trial.suggest_categorical('max_features', ['log2', 'sqrt'])

    # define Random Forest classifier
    RF = RandomForestClassifier(n_estimators=config.n_estimators,
                                max_depth=config.max_depth,
                                min_samples_leaf=config.min_samples_leaf,
                                max_features=config.max_features)
    # load dataset
    X_train, _, y_train, _ = load_data(config)

    # define k-fold cross validation
    cv = ShuffleSplit(n_splits=config.CV, test_size=config.split)
    # define scores: maximize accuracy
    scores = model_selection.cross_val_score(RF, X_train, y_train.values.ravel(), n_jobs=-1, cv=cv, scoring='roc_auc')

    return np.mean(scores)


def max_trial_callback(study, trial):
    n_complete = len([t for t in study.trials if
                      t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    if n_complete > config.number_trial - 1:
        study.stop()


def train(config, X_train, y_train):
    """training model

    Args:
        config:
        train_sliding_feature (pd.DataFrame): input feature space
        train_sliding_label (pd.DataFrame): input label
    """
    rf = RandomForestClassifier(n_estimators=config.n_estimators,
                                max_depth=config.max_depth,
                                max_features=config.max_features,
                                min_samples_leaf=config.min_samples_leaf)
    rf.fit(X_train, y_train)
    return rf


def RF_train(config):
    """ Training procedure in Random Forest.

    Args:
        config (_type_): configuration of parameters.
    """
    x_train, x_test, y_train, y_test = load_data(config)

    if config.mode == "train":
        train_RF_model = train(config, x_train, y_train)
        save(config, train_RF_model, 'train')
        print(f'retrained model has been saved.')

    elif config.mode == "test":
        saved_RF_model = load(config, 'train')
        pred_test = saved_RF_model.predict(x_test)
        prob_test = saved_RF_model.predict_proba(x_test)
        prob_test = np.max(prob_test, axis=1)
        report = classification_report(y_test, pred_test, output_dict=True)

        print(report)
        print("Test finished.")

        # save prediction labels to csv file
        # y_test['prediction'] = pred_test
        # y_test['probability'] = prob_test
        # y_test.to_csv(f'{config.save_path}/test_prediction.csv')

        # save test metrics to csv file
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'{config.save_path}/test_metrics.csv')


def save(config, model, flag):
    joblib.dump(model, f"{config.save_path}/retrained_model.joblib")


def load(config, flag):
    load_model = joblib.load(f"{config.save_path}/retrained_model.joblib")
    return load_model


if __name__ == '__main__':
    config = base_parser()
    if config.mode == "tune_hyperparas":
        if config.GPU != '-1':
            config.GPU_print = [int(config.GPU.split(',')[0])]
            os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
            config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
        else:
            config.GPU = False

        np.random.seed(config.seed)

        config.save_path = os.path.join(config.save_path, f'RF_models_forecast_PFS_{config.PFS_threshold}months')
        config.save_path_reports = os.path.join(config.save_path, 'reports')

        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.save_path_reports, exist_ok=True)

        study = optuna.create_study(study_name=config.exp_name, sampler=optuna.samplers.TPESampler(),
                                    direction='maximize')

        study.optimize(lambda trial: objective(trial, config), n_jobs=1, callbacks=[max_trial_callback])

        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('Avg accuracy', trial.value)

        name_csv = os.path.join(config.save_path, 'Best_hyperparameters.csv')

        print('  Params: ')
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        trials_df = study.trials_dataframe()
        trials_df.to_csv(f'{config.save_path_reports}/CV_results.csv')
        trials_df.head()

        dic = dict(trial.params)
        dic['value'] = trial.value
        df = pd.DataFrame.from_dict(data=dic, orient='index').to_csv(name_csv, header=False)

    elif config.mode == 'train' or 'test':
        RF_train(config)



