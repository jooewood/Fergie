#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# @author: zdx
# =============================================================================

import os
from rdkit import Chem
import pandas as pd
from utils import validity_filter, add_InchI, save_file
from tokenizer import encode_test
from parallelize_apply import parallelize_dataframe
import numpy as np
import re

def pro_sm(smiles):
    try:
        smiles = str(smiles)
        #print('smiles is:',smiles)
        smiles = re.sub('\[\d+', '[', smiles)
        # if it doesn't contain carbon atom, it cannot be drug-like molecule, 
        # just remove
        if smiles.count('C') + smiles.count('c') < 2:
            return np.nan
        smiles = Chem.CanonSmiles(smiles, 0)
        # reserving the largest one if the molecule contains more than  one 
        # fragments,
        # which are seperated by '.'
        if '.' in smiles:
            frags = smiles.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smiles = frags[ix]
        return smiles
    except:
        return np.nan

def filter_sm(df):
    df['Filter_sm'] = df['SMILES'].apply(pro_sm)
    df['SMILES'] = df['Filter_sm']
    del df['Filter_sm']
    return df


def smiles_filter(df):
    print("Start to smiles filter")
    df = parallelize_dataframe(df, filter_sm)
    df = df.dropna()
    return df


def add_index(df):
    l = len(str(len(df)))
    nums = []
    for i in range(len(df)):
        nums.append(str(i).zfill(l))
    df['ID'] = nums
    return df

def InChI_filter(df):
    # Use InChI to unique
    df = parallelize_dataframe(df, add_InchI)
    df = df.dropna()                   #delete  rows with null values
    df.drop_duplicates(subset=['InChI'], inplace=True)  
    return df

def token_filter(df):
    res_df, pro_df, l_fail = encode_test(df)
    return res_df

def sdf2df(file_name):
    #Chem.SDMolSupperlier can read smiles mol file and sdf file
    mols = [ mol for mol in Chem.SDMolSupplier( file_name ) ]  
    
    smiles = []
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        smiles.append(smi)
    df = pd.DataFrame({
        'SMILES':smiles,
        'ID': 1
        })
    return df

def file_name(file):
    if '.gz' == file[-3:]:
        file = file.replace('.gz', '')
    name = os.path.splitext(os.path.basename(file))[0]
    return name

def file_format(file):
    if '.gz' in file:
        file = file.replace('.gz', '')
    format_ = os.path.splitext(os.path.basename(file))[1].split('.')[-1]
    if format_ == 'ism':
        format_ = 'smi'
    return format_

def preprocess(input_file, out_dir=None,out_file=None):
    if '_preprocess' in input_file:
        return input_file
    informat = file_format(input_file)
    inname = file_name(input_file)
    if out_file is None:
        if out_dir is None:
            out_dir = os.path.dirname(input_file)
        out_file = os.path.join(out_dir, inname + "_preprocess.csv" )
    if os.path.exists(out_file):
        return out_file
    if informat == "sdf":
        df = sdf2df(input_file)
    if informat == "csv":
        df = pd.read_csv(input_file)
    if 'ID' not in df.columns:
        f = add_index(df)
    df = df[['ID', 'SMILES']]
    if "plogP" not in df.columns:
        df["plogP"] = 1.1
    # Remove invalidity SMILES
    df = validity_filter(df)
    # Remove duplicates molecules
    df = InChI_filter(df)
    # Remove unformatted SMILES string 
    df = smiles_filter(df)
    
    df = token_filter(df)
    if 'ID' not in df.columns:
        df = add_index(df)
    save_file(df, out_file)
    print(f"Molecules in {input_file} has been preprocessed, and the "
          f"preprocessed molecules has been saved into {out_file}.")
    return out_file    

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-f", "--input_file", type=str, required=True, 
        nargs="+", help="Files which will be preprocessed.")
    ap.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Folder path to save the preprocessed molecules.")
    args = ap.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for preprocess_file in args.preprocess_file:
        preprocess(
            preprocess_file = preprocess_file, 
            out_dir = args.output_dir
        )