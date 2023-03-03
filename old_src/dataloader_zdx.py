#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np

path_active = '/home/tensorflow/deepdrug_gan/data_test_for__dataload/active.csv'
path_zinc = '/home/tensorflow/deepdrug_gan/data_test_for__dataload/zinc.csv'
path_kinase = '/home/tensorflow/deepdrug_gan/data_test_for__dataload/kinase.csv'
path_non_kinase = '/home/tensorflow/deepdrug_gan/data_test_for__dataload/non_kinase.csv'

sources=[
    # 100 active
    {'path':path_active,
     'smiles': 'SMILES',
     'prob': 1,
     'plogP' : 'plogP',
    },
    # 60 zinc
    {'path':path_zinc,
     'smiles': 'SMILES',
     'prob': 0.6,
     'plogP' : 'plogP',
    },
    # 20 kinase
    {'path':path_kinase,
     'smiles': 'SMILES',
     'prob': 0.2,
     'plogP' : 'plogP',
    },
    # 20 non kinase
    {'path':path_non_kinase,
     'smiles': 'SMILES',
     'prob': 0.2,
     'plogP' : 'plogP',
    },
    ]
props=['plogP']
with_missings=False

class MolecularDataset(Dataset):
    def __init__(self, sources=[], props=['logIC50', 'BFL', 'pipeline'],
                 with_missings=False):
        num_sources = len(sources)

        source_smiles = []
        source_props = []
        source_missings = []
        source_probs = []

        with_missings = with_missings

        length = 0
        for source_descr in sources:
            cur_df = pd.read_csv(source_descr['path'])
            cur_smiles = list(cur_df[source_descr['smiles']].values)

            cur_props = torch.zeros(len(cur_smiles), len(props)).float()
            cur_missings = torch.zeros(len(cur_smiles), len(props)).long()

            for i, prop in enumerate(props):
                if prop in source_descr:
                    if isinstance(source_descr[prop], str):
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr[prop]].values)
                    else:
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr['smiles']].map(
                                source_descr[prop]).values)
                else:
                    cur_missings[:, i] = 1

            source_smiles.append(cur_smiles)
            source_props.append(cur_props)
            source_missings.append(cur_missings)
            source_probs.append(source_descr['prob'])

            length = max(length,
                           int(len(cur_smiles) / source_descr['prob']))

        source_probs = np.array(source_probs).astype(np.float)

        source_probs /= source_probs.sum()

    def __len__(self):
        return length

    def __getitem__(self, idx):
        trial = np.random.random()

        s = 0
        for i in range(num_sources):
            if (trial >= s) and (trial <= s + source_probs[i]):
                bin_len = len(source_smiles[i])
                sm = source_smiles[i][idx % bin_len]

                props = source_props[i][idx % bin_len]
                miss = source_missings[i][idx % bin_len]

                if with_missings:
                    return sm, torch.concat([props, miss])
                else:
                    return sm, props

            s += source_probs[i]