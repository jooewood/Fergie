#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from rdkit import Chem
from moses.metrics.utils import get_mol
import pandas as pd
import pickle
from rdkit.Chem import Draw
f = open('A_frag_hpk1', 'rb')
A_frag_hpk1 = pickle.load(f)

#f = open('A_frag_hpk1_diff_score_dict', 'rb')
#A_frag_hpk1_diff_score_dict = pickle.load(f)

f = open('A_frag_hpk1_same_score_dict', 'rb')
A_frag_hpk1_same_score_dict = pickle.load(f)

def substruct_match(smi, seq):
    segment = Chem.MolFromSmiles(seq)
    whole_smiles = Chem.MolFromSmiles(smi)
    return whole_smiles.HasSubstructMatch(segment)

## second version
def add_score(smiles, score_dict=A_frag_hpk1_same_score_dict):
    sum_score = 0
    for frag in A_frag_hpk1:
        if substruct_match(smiles, frag):
            sum_score += score_dict[frag]
    return sum_score

def reward_fn(smiles, punishment=-1):
    mol = get_mol(smiles)
    if mol is None:
        return punishment
    sum_score = add_score(smiles)
    if sum_score == 0:
        return punishment
    else:
        return sum_score
    
#Draw.MolsToGridImage([get_mol(s) for s in frag_list])
# creare dict
#score_df = pd.read_csv('/home/tensorflow/Desktop/理想/CloudStation/A_fragment.csv')
#smiles = list(score_df['mol'])
#scores = list(score_df['score'])
#A_frag_hpk1_same_score_dict = {}
#for i in range(len(smiles)):
#    tmp_smile = smiles[i]
#    tmp_score = 1
#    A_frag_hpk1_same_score_dict.update({tmp_smile:tmp_score})
#f = open('A_frag_hpk1_same_score_dict', 'wb')
#pickle.dump(A_frag_hpk1_same_score_dict, file=f)

## first version 
# =============================================================================
# def reward_fn(smiles, punishment=-1, reward=1):
#     mol = get_mol(smiles)
#     if mol is None:
#         return punishment
#     for frag in A_frag_hpk1:
#         if substruct_match(smiles, frag):
#             return reward
#     return punishment
# =============================================================================