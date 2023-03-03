#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, PandasTools
from utils import drop_duplicates_between_two_dataframes, file_name_format
from pycaret.classification import setup, compare_models, tune_model,\
    finalize_model, predict_model, save_model, load_model, create_model

def get_fingerprint(df):
    print('Getting fingerprint ...')
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    fps = []
    for mol in df.ROMol:
        fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        fps.append(fp)
    fps = np.array(fps)
    X = pd.DataFrame(fps)
    return X

def train_qsar_pycaret(active_file, decoy_file, out_dir, downsample=False, 
                       session_id=123, fold=3, ignore_low_variance=True):
    print("Training ...")
    """
    downsample=True
    session_id=123
    fold=3
    ignore_low_variance=True
    """
    active = pd.read_csv(active_file)
    decoy = pd.read_csv(decoy_file)
    decoy = drop_duplicates_between_two_dataframes(decoy, active)
    if downsample:
        decoy = decoy.sample(n=len(active))
    active = active.copy(); active['LABEL'] = 1
    active = active[['SMILES', 'LABEL']]
    decoy['LABEL'] = 0
    decoy = decoy[['SMILES', 'LABEL']]
    df = pd.concat([active, decoy])
    df.reset_index(drop=True, inplace=True)
    
    X = get_fingerprint(df)
    X['Y'] = df['LABEL'].values
    
    exp_clf101 = setup(data = X, target = 'Y', session_id=session_id, fold=fold, 
                       html=False, silent=True, ignore_low_variance=ignore_low_variance) 
    
    lr = create_model('lr')
    
    best_model = compare_models()
    tuned_best_model = tune_model(best_model)
    final_model = finalize_model(tuned_best_model)
    out_file = os.path.join(out_dir, 'qsar_best_ml_model')
    save_model(final_model, out_file)
    return out_file, final_model
    
    
def predict_qsar_pycaret(test_file, out_file=None, out_dir=None, 
                         model_file=None, model=None):
    print("Predicting ...")
    if out_file is None:
        name, _ = file_name_format(test_file)
        if out_dir is None:
            out_dir = os.path.dirname(test_file)
        out_file = os.path.join(out_dir, name+'_scored.csv')
    if os.path.exists(out_file):
        print("The file has been predicted")
        return out_file

    if model is None:
        if model_file is None:
            print("Do not have input model, please check!")
            return
        if '.pkl' in model_file:
            model_file = model_file.replace('.pkl', '')
        
        model = load_model(model_file)
    df = pd.read_csv(test_file)
    X_test = get_fingerprint(df)
    del df['ROMol']
    unseen_predictions = predict_model(model, data=X_test)
    df['score'] = unseen_predictions.Score
    df.sort_values('score', ascending=False, inplace=True)

    df.to_csv(out_file, index=False)
    return out_file
    
def train_predict_pycaret(active_file, decoy_file, test_file, out_dir, 
                          session_id=123, fold=3):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_file, model = train_qsar_pycaret(active_file, decoy_file, out_dir, 
                                    session_id, fold)
    test_predicted_file = predict_qsar_pycaret(test_file, out_dir=out_dir, 
                                               model=model)
    return model_file, test_predicted_file