#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd
from tokenizer import encode_test
from rdkit import Chem


df = pd.read_csv('/home/tensorflow/Downloads/hpk1_dat/zinc_lead_like_2019-09-25.csv')

sm_list = list(df['SMILES'])
res_df, l_fail = encode_test(sm_list)

df.drop(df.index[indexs],inplace=True)
df.to_csv('hpk1_other_data_plogp.csv', index=False)

size = 10000000
count = 0
for i in range(0,len(df.iloc[:,0]), size):
    # i = 0
    df_tmp = df[i:i+size]
    path = 'data_split/zinc_' + str(count)
    count += 1
    df_tmp.to_csv(path, index=False)
    
"""
    '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27
"""