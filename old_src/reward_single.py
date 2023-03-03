#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

from moses.metrics import mol_passes_filters
from moses.metrics.utils import get_mol

from rdkit.Chem import DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem

def smiles_to_ecfp4(smiles):
	'''Use SMILES to calculate ECFP4.'''
	try:
		mol = MolFromSmiles(smiles)
		return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
	except:
		return None

def tanimoto(fp1, fp2): 
	'''Compute Tanimoto similarity between two fingerprints'''
	return DataStructs.FingerprintSimilarity(fp1,fp2)
smiles = 'COc1cccc(F)c1-c1ncc2[nH]nc(-c3ccccc3)c2n1'
fp1 = smiles_to_ecfp4(smiles)

def reward_fn(smiles, default=-1):
    mol = get_mol(smiles)
    if mol is None:
        return default
#    if not mol_passes_filters(mol):
#        return default
    fp2 = smiles_to_ecfp4(smiles)
    score = tanimoto(fp1, fp2)
#    if score > 0.8:
#        reward = 1
#    else:
#        reward = -1
    return score
    
