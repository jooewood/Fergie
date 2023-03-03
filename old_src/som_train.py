#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from minisom import MiniSom
import math
import pickle
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from parallelize_apply import parallelize_dataframe
from numpy import sum as npsum
from sklearn.model_selection import StratifiedKFold, KFold


## som model
class SOM(object):
    def __init__(self, sigma=1, learning_rate=0.5, neighborhood_function='bubble'):
        self.sigma = sigma
        self.lr = learning_rate
        self.nei_fn = neighborhood_function
    
    def smiles_to_ecfp4(smiles):
        mol = MolFromSmiles(smiles)
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    def fit(self, X_SMILES, y_train):
        #  X_SMILES = tmp_x
        df = pd.DataFrame({'SMILES':X_SMILES})
        X_train = df['SMILES'].apply(smiles_to_ecfp4)
        X = X_train.values
        X_copy = []
        for item in X:
            X_copy.append(item)
        X_train = np.array(X_copy)
        del X_copy
        
        N = X_train.shape[0]  # number of samples
        M = X_train.shape[1]  # number of features
        size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # size of output layer
        max_iter = N
        self.som = MiniSom(size, size, M, sigma=self.sigma, learning_rate=self.lr, 
                      neighborhood_function=self.nei_fn)
        self.som.random_weights_init(X_train)
        self.som.train_batch(X_train, max_iter, verbose=False)
        self.winmap = self.som.labels_map(X_train, y_train)
        
    def predict(self, X_SMILES):
        def smiles_to_feature(SMILES):
            mol = Chem.MolFromSmiles(SMILES)
            feature = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            return feature
        def p_spec_label(win_position, winmap, label=1):
            if len(winmap[win_position])==2:
                node = winmap[win_position].most_common()
                d = {}
                d[node[0][0]] = node[0][1]
                d[node[1][0]] = node[1][1]
                p = d[label] / sum(d.values())
                return p
            elif len(winmap[win_position])==1:
                node = winmap[win_position].most_common()
                if node[0][0]==label:
                    return 1
                else:
                    return 0
            else:
                return 0
        def reward(smiles, som, winmap):
            feature = smiles_to_feature(smiles)
            win_position = som.winner(feature)
            if win_position in winmap:
                return p_spec_label(win_position, winmap)
            else:
                return 0
        def score(som, data, winmap):
            result = []
            for d in data:
                result.append(reward(d, som, winmap))
            return result
        y_pred_c = score(self.som, X_SMILES, self.winmap)
        return y_pred_c
    
## convert2som_train
all_data = pd.read_csv('/home/tensorflow/Downloads/SOM/data_for_test_reward.csv')                              
all_data['ID'] = list(range(len(all_data[:])))
X = all_data['CANONICAL_SMILES'].values
y = all_data['LABEL'].values

folds = StratifiedKFold(5).split(X, y)
score = np.zeros(y.shape)
for i, (trained, valided) in enumerate(folds):
    trained = list(trained)
    valided = list(valided)
    model = SOM()
    model.fit(X[trained], y[trained])
    score[valided] = model.predict(X[valided])

df = pd.DataFrame({'SCORE':score})
df['CANONICAL_SMILES'] = all_data['CANONICAL_SMILES'].values
df['LABEL'] = all_data['LABEL'].values

df.to_csv('/home/tensorflow/Desktop/som_score.csv', index=False)
