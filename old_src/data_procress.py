#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd

data_names = ['hair_dryer', 'microwave', 'pacifier']
for data_name in data_names:
    path = '../data/raw/{}.tsv'.format(data_name)
    df = pd.read_table(path)
    path = '../data/{}.csv'.format(data_name)
    df.to_csv(path, index=False)
