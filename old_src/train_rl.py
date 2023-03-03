#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import gentrl
import encoder
import decoder
import pandas as pd
import numpy as np
import torch
import time
from rdkit.Chem import Draw
#from one_frag_reward import reward_fn
#from segment_reward import reward_fn
#from reward import reward_fn
#from reward_single import reward_fn
from hpk1_reward_RF import reward_fn
import matplotlib.pyplot as plt
torch.cuda.set_device(3)

path = 'pretrain_4_source_codesize_50_batchsize_1600_epoch_100' # model for RL
num_iterations = 50000

tmp = path.split('codesize_')[1]
latent_size = int(tmp.split('_')[0])
time_start=time.time()
enc = encoder.RNNEncoder(latent_size=latent_size)
dec = decoder.DilConvDecoder(latent_input_size=latent_size)
model = gentrl.GENTRL(enc, dec, latent_size * [('c', 10)], [('c', 10)], latent_size=latent_size, tt_int=30, beta=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#model.cuda();
model_path = '../{}/'.format(path)
model.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#model.cuda();

from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_n_rings, get_mol

from moses.utils import disable_rdkit_log
disable_rdkit_log()

time_start=time.time()
## model after RL
NO = -1
rl_outpath = '../after_rl_{}_iter{}'.format(path, num_iterations)
base = rl_outpath
if not os.path.exists(rl_outpath):
    os.system('mkdir -p {}'.format(rl_outpath))
else:
    while(os.path.exists(rl_outpath)):
        NO += 1
        rl_outpath = base + '_%d' % NO
    os.system('mkdir -p {}'.format(rl_outpath))
performance_path = '{}/performance'.format(rl_outpath)
df = model.train_as_rl(reward_fn, performance_path, num_iterations=num_iterations)
model.save('{}/'.format(rl_outpath))
time_end=time.time()

## draw picture
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
N = len(df['mean_reward'])
values = df['mean_reward']
index = range(N)
plt.bar(index, values, label="rainfall", color="#87CEFA")
plt.xlabel('iterations')
plt.ylabel('mean reward')
plt.title('time cost: %d s' % (time_end-time_start))
plt.savefig('{}/reward.jpg'.format(rl_outpath)) 
plt.close()
print('time cost',time_end-time_start,'s')

generated = []
while len(generated) < 200:
    sampled = model.sample(100, latent_size)
    sampled_valid = [s for s in sampled if get_mol(s)]
    generated += sampled_valid
sampled_valid = generated[0:200]
img = Draw.MolsToGridImage([get_mol(s) for s in sampled_valid], molsPerRow=10, subImgSize=(300,300))
img.save('{}/gen_mol.jpg'.format(rl_outpath))