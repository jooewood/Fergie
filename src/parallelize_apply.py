#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing
from functools import partial

CPUs = multiprocessing.cpu_count()

num_partitions = CPUs # number of partitions to split dataframe
num_cores = CPUs  # number of cores on your machine
#iris = pd.DataFrame(sns.load_dataset('iris'))

def parallelize_dataframe(df, func, **kwargs):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# =============================================================================
# def multiply_columns(data):
#     data['length_of_word'] = data['species'].apply(lambda x: len(x))
#     return data
# 
# iris = parallelize_dataframe(iris, multiply_columns)
# =============================================================================
