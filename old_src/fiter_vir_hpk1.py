#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd
from hpk1_reward_RF import reward_fn
from parallelize_apply import parallelize_dataframe
path = '/home/tensorflow/Desktop/理想/CloudStation/A6_library.csv'

data = pd.read_csv(path)
data.to_csv('/home/tensorflow/Desktop/理想/CloudStation/A6_library.csv', index=False)
def multiply_columns(data):
    data['score'] = data['SMILES'].apply(reward_fn)
    return data
