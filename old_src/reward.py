#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
from minisom import MiniSom
import pickle
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import numpy as np
from multiprocessing import Pool

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])
num_partitions = 10 #number of partitions to split dataframe
num_cores = 10 #number of cores on your machine
#iris = pd.DataFrame(sns.load_dataset('iris'))

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

#common_kinase = pd.read_csv("ligand_common_kinase.csv")
#non_kinase = pd.read_csv("ligand_non_kinase.csv")

## general som model
with open('kinase_vs_nonkinase.model', 'rb') as infile:
    gsom = pickle.load(infile)
with open('kinase_vs_nonkinase.winmap', 'rb') as infile:
    gwinmap = pickle.load(infile)

## specific som model
with open('hpk1_som.model', 'rb') as infile:
    ssom = pickle.load(infile)
with open('hpk1_som.winmap', 'rb') as infile:
    swinmap = pickle.load(infile)
    
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
## general
def general_som(smiles, gsom=gsom, gwinmap=gwinmap):
    "label is the target string which we concerned: common-kinase or non-kinase or MAP4K1"
    feature = smiles_to_feature(smiles)
    win_position = gsom.winner(feature)
    if win_position in gwinmap:
        return p_spec_label(win_position, gwinmap)
    else:
        return 0

## specific
def specific_som(smiles, gsom=ssom, gwinmap=swinmap):
    "label is the target string which we concerned: common-kinase or non-kinase or MAP4K1"
    feature = smiles_to_feature(smiles)
    win_position = gsom.winner(feature)
    if win_position in gwinmap:
        return p_spec_label(win_position, gwinmap)
    else:
        return 0

# smiles = 'CC(C)CN1CCCC(NC(=O)c2cccc(C)c2)CCSCCC1'
def reward_fn(smiles, default=-1):
    # data = common_kinase
    mol = get_mol(smiles)
    if mol is None:
        return default
#    if not mol_passes_filters(mol):
#        return default
#    if logP(mol) > 5 or logP(mol) < 2:
#        return default
    #scores = general_som(smiles)
    scores = specific_som(smiles)
    #scores = logP(mol) - SA(mol) - get_num_rings_6(mol)
    #scores = gscores + sscores
    return scores
