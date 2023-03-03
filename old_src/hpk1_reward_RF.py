#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# @author: zdx
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from moses.metrics.utils import get_n_rings, get_mol
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools

clf_path = 'RF_hpk1_model.pkg'
clf = joblib.load(clf_path)

# smiles = 'CC(C)CN1CCCC(NC(=O)c2cccc(C)c2)CCSCCC1'
def reward_fn(smiles, default=-1):
    mol = get_mol(smiles)
    if mol is None:
        return default
    feature = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
    feature = np.reshape(feature, [1, -1])
    scores = clf.predict_proba(feature)[0,1]
    return scores

""" train RF
input_path = '/home/tensorflow/Downloads/SOM/data_for_test_reward.csv'
out = 'RF_hpk1_model'
df = pd.read_csv(input_path)
PandasTools.AddMoleculeColumnToFrame(df, "CANONICAL_SMILES")
# df.columns
fps = []
for mol in df.ROMol:
    fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)]
    fps.append(fp)

#
fps = np.array(fps)
fps.shape 
targets = df.LABEL
targets.shape
X = fps
y = targets

================ grid search ==================
grid_param = {
    'n_estimators': [550, 600, 700, 800, 900, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [False]
}
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
gd_sr.fit(X, y)
best_parameters = gd_sr.best_params_
print(best_parameters)
===============================================

============ Five cross -validation ===========
folds = StratifiedKFold(5).split(X, y)
y_pred = np.zeros(y.shape)
for i, (trained, valided) in enumerate(folds):
    trained = list(trained)
    valided = list(valided)
    model = RandomForestClassifier(bootstrap=False, criterion='gini', n_estimators=600, n_jobs=-1)
    model.fit(X[trained], y[trained])
    y_pred[valided] = model.predict_proba(X[valided])[:, 1]

fpr, tpr, ths = metrics.roc_curve(y, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
===============================================
model = RandomForestClassifier(bootstrap=False, criterion='gini', n_estimators=600, n_jobs=1)
model.fit(X, y)
joblib.dump(model, out+'.pkg')
"""
