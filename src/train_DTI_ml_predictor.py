#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from moses.metrics.utils import get_n_rings, get_mol
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from parallelize_apply import parallelize_dataframe
import math

def grid_search(clf, grid_param, X, y):
    gs = GridSearchCV(estimator = clf,
                      param_grid = grid_param,
                      scoring = 'roc_auc',
                      cv = 5,
                      n_jobs = 40)
    gs.fit(X, y)
    params = gs.best_params_
    return params
## models
def LR(X, y):
    clf = LogisticRegression(n_jobs=1)
    grid_param  = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'penalty': ['l1', 'l2'],
                   'random_state': list(range(10)),
                   'max_iter': list(range(0,500, 50)),
                   #'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                   }
    params = grid_search(clf, grid_param, X, y)
    return LogisticRegression(C = params['C'],
                              penalty = params['penalty'],
                              random_state = params['random_state'],
                              max_iter = params['max_iter'], solver = 'lbfgs', n_jobs=1)
def DT(X, y):
    max_depth = list(range(10, 110, 5))
    max_depth.append(None)
    grid_param  = {'max_features': ['auto', 'sqrt', 'log2', None],
                   'max_depth': max_depth,
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'random_state': list(range(0, 50, 5))
                   }
    clf = DecisionTreeClassifier()
    params = grid_search(clf, grid_param, X, y)
    return DecisionTreeClassifier(max_features = params['max_features'],
                                  max_depth = params['max_depth'],
                                  min_samples_split = params['min_samples_split'],
                                  min_samples_leaf = params['min_samples_leaf'],
                                  random_state=params['random_state'])
def SVM(X, y):
    clf = SVC(probability=True)
    C = [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]
    gamma = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    grid_param = {'C': C,
                  'kernel': ['rbf'], #'poly', 'precomputed', 'linear', 'rbf', 'sigmoid'
                  'gamma': gamma}
    params = grid_search(clf, grid_param, X, y)
    return SVC(probability=True, 
               kernel = params['kernel'],
               C = params['C'], 
               gamma = params['gamma'],
               random_state = 42)

def RF(X, y):
    n_estimators = list(range(20, 1000, 10))
    max_features = ['auto', 'sqrt', 'log2', None]
    max_depth = list(range(10, 110, 5))
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier(n_jobs=1)
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions=random_grid,
                                   n_iter=100,cv=5,verbose=2,random_state=42, 
                                   n_jobs = 40)
    rf_random.fit(X, y)
    params = rf_random.best_params_
    return RandomForestClassifier(bootstrap=params['bootstrap'],
                                  max_depth=params['max_depth'],
                                  max_features=params['max_features'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  n_estimators=params['n_estimators'], 
                                  random_state=42,
                                  n_jobs = 1)
def KNN(X, y):
    n_neighbors = list(range(1,10))
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = list(range(5, 50, 5))
    p = [1, 2]
    clf = KNeighborsClassifier(n_jobs=1)
    grid_param = {'n_neighbors': n_neighbors,
                              'algorithm': algorithm,
                              'leaf_size': leaf_size,
                              'p': p}
    params = grid_search(clf, grid_param, X, y)
    return KNeighborsClassifier(n_neighbors = params['n_neighbors'],
                                algorithm = params['algorithm'],
                                leaf_size = params['leaf_size'],
                                p = params['p'],
                                n_jobs = 1)


def get_pos_neg(target_name, active_file, negative_file):
    active_df = pd.read_csv(active_file); active_df['LABEL'] = 1
    active_df = active_df[['ID', 'SMILES', 'LABEL']]
    active_df = active_df.drop_duplicates(['ID'])
    
    decoy = pd.read_csv(negative_file); decoy['LABEL'] = 0
    decoy = decoy[['ID', 'SMILES', 'LABEL']]
    decoy = decoy.reindex(np.random.permutation(decoy.index))
    decoy = decoy.reset_index(drop=True)
    tmp = decoy.copy()
    tmp = tmp.append(active_df)
    tmp = tmp.append(active_df)
    decoy = tmp.drop_duplicates(['ID'], keep=False)
    decoy_df = decoy.sample(n=len(active_df)*2, replace=False)
    decoy_df = decoy_df.reset_index(drop=True)
    return active_df, decoy_df

target_name = 'shp2'
active_file = '/y/Aurora/Fergie/data/preprocessed/shp2/shp2_preprocess.csv'
negative_file = '/y/Aurora/Fergie/data/preprocessed/shp2/TPP_preprocess.csv'

active_df, decoy_df = get_pos_neg(target_name, active_file, negative_file)


def smiles_to_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
def judge_ECFP_duplicate(sm, df=active_df):
    f2 = smiles_to_ecfp4(sm)
    def Tc_compute(sm1, f2=f2):
        f1 = smiles_to_ecfp4(sm1)
        Tc = DataStructs.FingerprintSimilarity(f1,f2)
        return Tc
    df['Tc'] = df['SMILES'].apply(Tc_compute)
    if 1 in df['Tc'].values:
        return "F"
    else:
        return "T"
def multi_Tc(data):
    data['remain'] = data['SMILES'].apply(judge_ECFP_duplicate)
    return data
decoy_df = parallelize_dataframe(decoy_df, multi_Tc)
decoy_df = decoy_df[decoy_df['remain']=="T"]
decoy_df = decoy_df[0:len(active_df)]
del decoy_df['remain']
df = pd.concat([active_df, decoy_df])

## train RF
best_auc = -1
best_model = None
best_length = 0

for bitss in [64, 128, 256, 512, 1024, 2048]: # 64, 128, 256, 512, 1024, 2048, 4096
    # bitss = 512
    task_folder = '../predictor_zoo/%s/%d' % (target_name, bitss)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)
    models_path = task_folder + '/models.pickle'
    fig_path = task_folder + '/performance.jpg'
    #df = parallelize_dataframe(df, SMILES2ECFP4)
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    fps = []
    for mol in df.ROMol:
        fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitss)]
        fps.append(fp)
    X = np.array(fps)
    y = df['LABEL'].values
    #============ Five cross -validation ===========
    legends = ['RF', 
               'SVM', 'KNN', 'NB', 'DT', 'LR']
    fig = plt.figure(figsize=(5, 5), dpi=600)
    ax1 = fig.add_subplot(111)
    lw = 1.5
    models = [RF(X, y),
              SVM(X, y),
              KNN(X, y),
              DT(X, y),
              GaussianNB(),
              LR(X, y)
              ]
    MODEL = models.copy()
    with open(models_path, 'wb') as f:
        pickle.dump(MODEL, f)
    auc_list = []
    for i, clf in enumerate(models):
        y_pred = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
        fpr, tpr, ths = metrics.roc_curve(y, y_pred)
        auc = metrics.auc(fpr, tpr)
        if auc > best_auc:
            best_length = bitss
            best_auc = auc
            best_model = clf
        ax1.plot(fpr, tpr, lw=lw, label=legends[i] + '(AUC=%.3f)' % auc)
        auc_list.append(auc)
    for i in range(1, 10):
        plt.plot([i * 0.1, i * 0.1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.plot([0, 1], [i * 0.1, i * 0.1], color='gray', lw=lw, linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right")
    plt.savefig(fig_path, dpi=600)
fps = []
for mol in df.ROMol:
    fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, best_length)]
    fps.append(fp)
X = np.array(fps)
best_model.fit(X, y)

if not os.path.exists('../predictor'):
    os.makedirs('../predictor')
joblib.dump(best_model, '../predictor/%s.pkg'%target_name)
with open('../predictor/%s.config'%target_name, 'w') as f:
    print("best model information:\n", best_model, file=f)
    print("best ECFP4 bits:", best_length, file=f)
    print("Best AUC:", best_auc, file=f)