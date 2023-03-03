#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os 
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
import time
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
import matplotlib.pyplot as plt
from parallelize_apply import parallelize_dataframe
import random
from tokenizer import encode_test, smiles_tokenizer
from sklearn.cluster import AgglomerativeClustering

# task_folder = '/home/tensorflow/mnt/zinc_drug_like_wget'

# files = glob.glob(task_folder+'/*')
# out_folder = os.path.dirname(task_folder)
# out_path = out_folder + '/zinc.txt'

# label = 0
# n = len(files)
# with open(out_path, 'w') as fout:
#     for num, file in enumerate(files):
#         print(n-num)
#         # file = files[0]
#         with open(file, 'r') as f:
#             cont = f.readlines()
#         if label!=0:
#             for i in cont[1:]:
#                 fout.write(i)
#         else:
#             for i in cont:
#                 fout.write(i)
#             label = 1
# with open('/home/tensorflow/mnt/ZINC.uri', 'r') as f:
#     content = f.readlines()
    
# def get_name(x):
#     # x = files[0]
#     x = x.split('/')[-1]
#     x = x.split('.txt')[0]
#     return x
# names = set(map(get_name, files))
# raw_names = set(map(get_name, content))
# diff = list(raw_names - names)
#========================= mol filter function  ===============================
def add_mol(smiles):
    mol = get_mol(smiles)
    if not mol == None:
        return mol
    else:
        return 'NA'
def multi_add_mol(data):
    data['ROMol'] = data['SMILES'].apply(add_mol)
    return data
#========================= four rings filter function =========================
def judge_whether_has_rings_4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    r = mol.GetRingInfo()
    if len([x for x in r.AtomRings() if len(x)==4]) > 0:
        return False
    else:
        return True
def multi_judge_whether_has_rings_4(data):
    data['retain'] = data['SMILES'].apply(judge_whether_has_rings_4)
    return data
def four_rings_filter(df):
    df = parallelize_dataframe(df, multi_judge_whether_has_rings_4)
    df = df[df['retain']==True]
    del df['retain']
    return df
#========================== MCF filter function ===============================
def add_MCF(data):
    data['MCF'] = data.ROMol.apply(mol_passes_filters)
    return data
#========================== pretrain filter ===================================
def clean_df_for_pretrain(df):
    res_df, pro_df, l_fail = encode_test(df)
    return res_df, pro_df, l_fail
#========================== atomic scaffold filter ============================
def multiply_get_scaffold(data):
    data['atomic_scaffold'] = data.ROMol.apply(GetScaffoldForMol)
    return data
#========================== custom molecule cluster ===========================
def sim(x, y): 
    return np.sum(np.equal(np.array(x), np.array(y)))/len(x)
# Method to calculate distances between all sample pairs
# from sklearn.metrics import pairwise_distances
# def sim_affinity(X):
#     return pairwise_distances(X, metric=sim)

# cluster = AgglomerativeClustering(n_clusters=5, affinity=sim_affinity, linkage='average')
# cluster.fit(X)
def chunk_filter(df):
    df = df[['zinc_id', 'smiles', 'mwt', 'logp']]
    df.columns = ['ID', 'SMILES', 'MW', 'logP']
    df['plogP'] = 1.1
    df = parallelize_dataframe(df, multi_add_mol)
    df = df[df['ROMol']!='NA']
    df = four_rings_filter(df)
    df = parallelize_dataframe(df, add_MCF)
    df = df[df['MCF']==True]
    df, _, _ = clean_df_for_pretrain(df)
    #df = parallelize_dataframe(df, multiply_get_scaffold)
    
    df = df[['ID', 'SMILES', 'MW', 'logP']]
    return df
    
# df = pd.read_table('/home/tensorflow/mnt/zinc.txt', nrows=1000)
df = pd.read_table('/home/tensorflow/mnt/zinc.txt', chunksize=100000)
chunk_list = []
for chunk in df:
    filtered_chunk = chunk_filter(chunk)
    chunk_list.append(filtered_chunk)
    
data = pd.concat(chunk_list)
data.to_csv('/home/tensorflow/mnt/zinc_clean.txt', index=False)


# df = df[['zinc_id', 'smiles', 'mwt', 'logp']]
# df = df.query('mwt>450 & mwt<650 & 2<=logp & logp<=5')
# df.to_csv('/home/tensorflow/mnt/zinc_whole.csv', index=False)