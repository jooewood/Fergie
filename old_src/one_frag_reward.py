#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd
from rdkit import Chem
from moses.metrics.utils import get_mol

df = pd.read_csv('/home/tensorflow/Desktop/理想/CloudStation/A_fragment - 3.csv')


frag_1 = df['mol'][0]
frag_2 = df['mol'][1]
frag_3 = df['mol'][2]

def substruct_match(smi, seq):
    segment = Chem.MolFromSmiles(seq)
    whole_smiles = Chem.MolFromSmiles(smi)
    return whole_smiles.HasSubstructMatch(segment)

def reward_fn(smiles, punishment=-1):
    mol = get_mol(smiles)
    if mol is None:
        return punishment
    if substruct_match(smiles, frag_3):
        return 1
    else:
        return punishment
    
# smiles = 'CC1OCCCC1n1cc2c(-c3ccc(N4CCN(C)CC4)cc3)n[nH]c2cc1=O'