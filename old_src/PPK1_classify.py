#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd

df_IC50 = pd.read_excel('/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1.xlsx', sheet_name='IC50')
df_100 = pd.read_excel('/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1.xlsx', sheet_name='Inhibition_at_100uM')

df_IC50 = df_IC50[['ID', 'SMILES', 'Label']]
df_100 = df_100[['ID', 'SMILES', 'Label']]

df_IC50['Label'] = df_IC50['Label'].replace(['Active', 'Inactive'], [1, 0])
df_100['Label'] = df_100['Label'].replace(['Active', 'Inactive'], [1, 0])

df_IC50.to_csv('/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1_IC50.csv', index=False)
df_100.to_csv('/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1_100.csv', index=False)

df_IC50_100 = pd.concat([df_IC50, df_100])
df_IC50_100.to_csv('/home/tensorflow/Desktop/work/CloudStation/now/PPK1/Classif/PaPPK1_IC50_100.csv', index=False)
