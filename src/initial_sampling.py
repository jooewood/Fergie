#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
@author: zdx
@Description: 
    This script is used to sample initial 30000 compounds.
================================================================================
"""
import matplotlib
matplotlib.use("Agg")

import os
from tqdm import tqdm
import gentrl
import encoder
import decoder
import pandas as pd
from moses.metrics.utils import get_mol
import torch
import time

"""
--------------------------------------------------------------------------------
Some functions.
--------------------------------------------------------------------------------
"""
def get_num_rings_6(mol):
    r = mol.GetRingInfo()
    return len([x for x in r.AtomRings() if len(x) > 6])

def penalized_logP(mol_or_smiles, masked=True, default=-5):
    mol = get_mol(mol_or_smiles)
    if mol is None:
        return default
    reward = logP(mol) - SA(mol) - get_num_rings_6(mol)
    if masked and not mol_passes_filters(mol):
        return default
    return reward
    
def gen_index(df, pre):
    l = len(str(len(df)))
    nums = []
    init_str = '0' * l
    
    for i in range(len(df)):
        num_str = str(i)
        num_str_len = len(num_str)
        num_str = init_str[:l-num_str_len] + num_str
        nums.append('_'.join([pre, num_str]))
    return nums

def get_num_rings(mol):
    r = mol.GetRingInfo()
    return len(r.AtomRings())

def filter_mol(smiles):
    mol = get_mol(smiles)
    if not mol == None:
        return True
    else:
        return False
        
def multiply_columns(data):
    data['label'] = data['SMILES'].apply(filter_mol)
    return data


"""
--------------------------------------------------------------------------------
Generate molecule from trained vae.
--------------------------------------------------------------------------------
"""

def init_sample(input_dir, # Location of trained vae model.
                output_dir, # Location to save the molecule generated.
                target_name, # Name of the drug target.
                latent_size, # Dimension of latent code. 
                num_sample, # Total number of molecules to generate.
                batch_size, # Number of molecules will be sampled in a batch.
                tt_int,
                beta,
                ):
    if input_dir[-1] == '/':
        input_dir = input_dir[:-1]
    l_token = os.path.split(input_dir)[1].split('_')[1]
    time_start=time.time()
    print(f'Setup model to generate molecules for {l_token} from {input_dir}, '
          f'time_start: {time_start}', flush=True)
    enc = encoder.RNNEncoder(latent_size=latent_size, l_token=l_token)
    dec = decoder.DilConvDecoder(latent_input_size=latent_size, split_len=l_token)
    model = gentrl.GENTRL(enc, dec, latent_size * [('c', 10)], [('c', 10)], 
        latent_size=latent_size, tt_int=tt_int, beta=beta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model.cuda()
    model.load(input_dir)
    model.to(device)
    #model.cuda()

    molecule_path = os.path.join(output_dir, 'molecule_generated')
    print(f'Outputting modules to {molecule_path}', flush=True)
    if not os.path.exists(molecule_path):
        os.makedirs(molecule_path)
    gen_valid = set()
    gen_all = []
    pbar = tqdm(total=num_sample)
    while len(gen_valid) < num_sample:
        len_before = len(gen_valid)
        sampled = model.sample(batch_size, latent_size)
        gen_all += sampled
        sampled = list(set(sampled))
        df_tmp = pd.DataFrame({'SMILES':sampled})
        df_tmp = multiply_columns(df_tmp)
        df_tmp = df_tmp[df_tmp['label']==True]
        sampled_valid = set(df_tmp.SMILES)
        gen_valid = sampled_valid | gen_valid
        if len(gen_valid) > num_sample:
            pbar.update(num_sample - len_before)
        else:
            pbar.update(len(gen_valid) - len_before)
    gen_valid = list(gen_valid)
    print("validity of generation:", len(gen_valid) / len(gen_all))
    
    df_valid = pd.DataFrame({'SMILES': gen_valid})
    df_valid['ID'] = gen_index(df_valid, target_name)
    df_valid = multiply_columns(df_valid)
    df_valid = df_valid[df_valid['label']==True]
    del df_valid['label']
    
    df_all = pd.DataFrame({'SMILES': gen_all})
    df_all['ID'] = gen_index(df_all, target_name)
    
    file_valid = os.path.join(molecule_path, "init_valid_sample_%d.csv" % num_sample)
    file_all = os.path.join(molecule_path, "init_all_sample_%d.csv" % num_sample)
    df_valid.to_csv(file_valid, index=False)
    df_all.to_csv(file_all, index=False)
    time_end=time.time()
    print(f'time cost {time_end-time_start}s', flush=True)
    return file_valid


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_dir", type=str, required=True, 
        help="Location of model saved.")
    ap.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Location to save the generated molecule.")
    ap.add_argument("-p", "--protein", type=str, required=True, 
        help="The name of the target.")
    ap.add_argument("-l", "--latent_size", type=int, default=50,
        help="Dimension of latent code size.")
    ap.add_argument("-n", "--num_sample", type=int, default=30000,
        help="Total number of molecule to sample.")
    ap.add_argument("-b", "--batch_size", type=int, default=2000, 
        help="Number of molecule to sample in a batch.")
    ap.add_argument("-t", "--tt_int", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.001)
    ap.add_argument('-g', '--gpuUSE', action='store_true', default=False,
        help='gpu use or not')
    ap.add_argument('--gpu', type=int, default=-1, 
        choices=list(range(torch.cuda.device_count())),help='Which GPU to use') 
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)  
    init_sample(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        latent_size = args.latent_size,
        num_sample = args.num_sample,
        batch_size = args.batch_size,
        target_name = args.protein,
        tt_int = args.tt_int,
        beta = args.beta
    )
