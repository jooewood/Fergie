#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE, RandomOverSampler)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)

name = 'IC50'

path = '/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1_%s.csv' % name
df = pd.read_csv(path)

pos = df.query('LABEL==1')
neg = df.query('LABEL==0')

n_pos = len(pos)
n_neg = len(neg)

models = (LinearSVC(random_state=0))

methods = (RandomOverSampler(random_state=0),
           RandomUnderSampler(random_state=0)
           )

over_sampling_methods = (RandomOverSampler(random_state=0),
                         SMOTE(random_state=0),
                         ADASYN(random_state=0),
                         BorderlineSMOTE(random_state=0, kind='borderline-1'),
                         BorderlineSMOTE(random_state=0, kind='borderline-2'),
                         KMeansSMOTE(random_state=0),
                         SVMSMOTE(random_state=0),
                         #SMOTENC(categorical_features=[0, 2], random_state=0)
                         )

under_sampling_methods = (ClusterCentroids(random_state=0),
                          RandomUnderSampler(random_state=0)
                          NearMiss(version=1),
                          NearMiss(version=2),
                          NearMiss(version=3),
                          EditedNearestNeighbours(),
                          RepeatedEditedNearestNeighbours(),
                          AllKNN(allow_minority=True),
                          CondensedNearestNeighbour(random_state=0),
                          OneSidedSelection(random_state=0),
                          NeighbourhoodCleaningRule(),
                          InstanceHardnessThreshold(random_state=0, 
                                                    estimator=LogisticRegression(solver='lbfgs',
                                                                                 multi_class='auto'))
                          )
ensemble
boosting_methods = 

with open
for bits in [64, 128, 256, 512, 1024]:
    for fold_id in range(5):
        X_train, y_train, X_test, y_test = get_data(df, bits, fold_id)
        for sampler in methods:
            for model in models:
                pipeline = make_pipeline(sampler, model)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                print(classification_report_imbalanced(y_test, y_pred))