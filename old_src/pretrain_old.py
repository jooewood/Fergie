#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import gentrl
import encoder
import decoder
import dataloader
from torch.utils.data import DataLoader
import glob
import torch
import pandas as pd
torch.cuda.set_device(3)
import numpy as np
import time
from pandas import DataFrame
import pickle
from parallelize_apply import parallelize_dataframe # parallelize apply function

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol
import matplotlib.pyplot as plt

def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])

def penalized_logP(mol_or_smiles, masked=False, default=-5):
    mol = get_mol(mol_or_smiles)
    if mol is None:
        return default
    reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
    if masked and not mol_passes_filters(mol):
        return default
    return reward

#os.system('wget https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv')

time_start=time.time()
BATCHSIZE = 2000
file_list = []
for i in range(10):
    path = 'data_split/zinc_' + str(i)
    file_list.append(path)
index = 0
for path in file_list:
    enc = encoder.RNNEncoder(latent_size=50)
    dec = decoder.DilConvDecoder(latent_input_size=50)
    model = gentrl.GENTRL(enc, dec, 50 * [('c', 10)], [('c', 10)], tt_int=30, beta=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model.cuda();
    if index==0:
        index = 1
    else:
        model.load('saved_gentrl/')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #model.cuda();

    md = dataloader.MolecularDataset(sources=[
        {'path':path,
         'smiles': 'SMILES',
         'prob': 1,
         'plogP' : 'plogP',
        }], 
        props=['plogP'])
    train_loader = DataLoader(md, batch_size=BATCHSIZE, shuffle=True, num_workers=24, drop_last=True)
    
    model.train_as_vaelp(train_loader, lr=1e-4)
    task_path = 'saved_gentrl/'
    if not os.path.exists(task_path):
        os.system('mkdir -p {}'.format(task_path))
    model.save('./{}/'.format(task_path))

time_end=time.time()
time_cost = round(time_end-time_start, 4)
print('{} time cost'.format(task_path),time_cost,'s')
