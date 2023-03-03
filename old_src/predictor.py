#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""


clf_path = 'RF_hpk1_model.pkg'
clf = joblib.load(clf_path)

# smiles = 'CC(C)CN1CCCC(NC(=O)c2cccc(C)c2)CCSCCC1'
def reward_fn(smiles, default=-1):
    mol = get_mol(smiles)
    if mol is None:
        return default
    feature = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
    feature = np.reshape(feature, [1, -1])
    scores = clf.predict_proba(feature)[0,1]
    return scores