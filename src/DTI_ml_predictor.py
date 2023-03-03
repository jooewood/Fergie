#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import joblib
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, PandasTools


def get_X(df, bitss):
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    fps = []
    for mol in df.ROMol:
        fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitss)]
        fps.append(fp)
    X = np.array(fps)
    return X


def reward_fn(X, model_file):
    clf = joblib.load(model_file)
    scores = clf.predict_proba(X)[:,1]
    return scores

def add_score(df, model_file, bits, predictor_name):
    X = get_X(df, bits)
    df['%sscore'%predictor_name] = reward_fn(X, model_file)
    df = df.sort_values(['%sscore'%predictor_name], ascending=False)
    del df["ROMol"]
    df = df.reset_index(drop=True)
    return df

def main(input_file, model_file, bits, predictor_name):
    df = pd.read_csv(input_file)
    df = add_score(df, model_file, bits, predictor_name)
    df.to_csv(input_file, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', "--input_file", required=True,
        help="File to add DTI score.")
    ap.add_argument('-m', '--model_file', required=True,
        help="Path of model file.")
    ap.add_argument('-b', '--bits', required=True, type=int,
        help="Number of bits.")
    ap.add_argument('-n', '--predictor_name', required=True,
        help="Name of predictor.")
    args = ap.parse_args()
    main(args.input_file, args.model_file, args.bits,
         args.predictor_name)
    
    
