#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from parallelize_apply import parallelize_dataframe
import joblib

predictor_name = 'under_sample/PPK1_IC50'
file_name = 'PaPPK1_100'
df = pd.read_csv('../dat/%s.csv' % file_name)
bits = 1024

def ranking_auc(df, mode='T'):
    if mode=='T':
        df = df.sort_values(by='score', ascending = True)
        df = df.reset_index(drop=True)
    else:
        df = df.sort_values(by='score', ascending = False)
        df = df.reset_index(drop=True)
    auc_roc, auc_prc=np.nan, np.nan
    l = len(df)
    a_pos = list(df[df['LABEL']==1].index)
    a_pos = np.array(a_pos) + 1
    ##Generate the seq of active cmpd
    a_seen = np.array(range(1,len(a_pos)+1))
    ##Calculate auc contribution of each active_cmpd
    d_seen = a_pos-a_seen
    ##Calculate AUC ROC
    a_total=len(a_pos)
    d_total = l - len(a_pos)
    contri_roc = d_seen/(a_total*d_total)
    auc_roc = 1.0 - np.sum(contri_roc)
    ##Calculate AUC PRC
    auc_prc = (1/a_total)*np.sum((1.0*a_seen)/a_pos)
    return auc_roc, auc_prc 

def enrichment(a_pos, total_cmpd_number, top):
    ##Calculate total/active cmpd number at top% 
    top_cmpd_number = np.around(total_cmpd_number*top)
    top_active_number = 0
    for a in a_pos:
        if a>top_cmpd_number: break
        top_active_number += 1
    ##Calculate EF
    total_active_number = len(a_pos)
    ef = (1.0*top_active_number/top_cmpd_number)*(total_cmpd_number/total_active_number)
    return ef

def enrichment_factor(df):
    l = len(df)
    a_pos = df[df['LABEL']==1].index
    a_pos = np.array(a_pos) + 1
    ef5 = enrichment(a_pos, l,  0.05)
    ef10 = enrichment(a_pos, l,  0.1)
    ef15 = enrichment(a_pos, l,  0.15)
    ef1 = enrichment(a_pos, l,  0.01)
    return ef1, ef5, ef10

PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
def reward_fn(mol, default=-1):
    clf_path = '../MODEL/%s.pkg' % predictor_name
    clf = joblib.load(clf_path)
    feature = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, bits))
    feature = np.reshape(feature, [1, -1])
    scores = clf.predict_proba(feature)[0,1]
    return scores
def multiply_columns(data):
    data['score'] = data.ROMol.apply(reward_fn)
    return data
df = parallelize_dataframe(df, multiply_columns)
df = df.sort_values(by='score', ascending = False)
del df['ROMol']

auc_roc, auc_prc = ranking_auc(df, mode='F')
ef1, ef2, ef3 = enrichment_factor(df)

print(auc_roc, auc_prc, ef1, ef2, ef3)

# from rdkit.Chem import Draw
# from rdkit import Chem
# from rdkit.Chem import BRICS
# m = Chem.MolFromSmiles('CCCOCc1cc(c2ncccc2)ccc1')
# Draw.MolToImage(m)
# res = list(BRICS.BRICSDecompose(m))
# resfrags = Chem.GetMolFrags(m,asMols=True)
# for i in res:
#     m = Chem.MolFromSmiles(i)
#     Draw.MolToImage(m)
