#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from moses.metrics import mol_passes_filters, QED, SA, logP
from moses.metrics.utils import get_mol
from parallelize_apply import parallelize_dataframe
from functools import partial
from rdkit.Chem import rdinchi

def file_format(file):
    if '.gz' in file:
        file = file.replace('.gz', '')
    format_ = os.path.splitext(os.path.basename(file))[1].split('.')[-1]
    if format_ == 'ism':
        format_ = 'smi'
    return format_

def file_name(file):
    if '.gz' == file[-3:]:
        file = file.replace('.gz', '')
    name = os.path.splitext(os.path.basename(file))[0]
    return name

def file_name_format(file):
    return file_name(file), file_format(file)


def drop_duplicates_between_two_dataframes(df1, df2):
    """
    Remove molecules which already existed in the second dataframe.
    """
    if 'InChI' not in df1.columns:
        df1 = add_InchI(df1)
    if 'InChI' not in df2.columns:
        df2 =  add_InchI(df2)
    df1.drop_duplicates('InChI', inplace=True)
    df2.drop_duplicates('InChI', inplace=True)
    df1 = df1.append(df2)
    df1 = df1.append(df2)
    print('df1+2*df2:', len(df1))
    df1.drop_duplicates(['InChI'], keep=False, inplace=True)
    print('df1:', len(df1))
    print('df2:', len(df2))
    return df1

"""
============================property functions=================================
"""
def judge_whether_has_rings_4(mol):
    r = mol.GetRingInfo()
    if len([x for x in r.AtomRings() if len(x)==4]) > 0:
        return False
    else:
        return True
    
def add_whether_have_4_rings(data):
    """four rings"""
    data['4rings'] = data['ROMol'].apply(judge_whether_has_rings_4)
    return data

def four_rings_filter(df):
    df = parallelize_dataframe(df, add_whether_have_4_rings)
    df = df[df['4rings']==True]
    del df['4rings']
    return df

def mol2InchI(mol):
    try:
        inchi, retcode, message, logs, aux = rdinchi.MolToInchi(mol)
        return inchi
    except:
        return np.nan

def add_InchI(df):
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, add_mol)
    df['InChI'] = df.ROMol.apply(mol2InchI)
    del df['ROMol']
    return df

def SMILES2MOL(smiles):
    try:
        mol = get_mol(smiles)
        if not mol == None and not mol == "":
            return mol
        else:
            return np.nan
    except:
        return np.nan
    
def MW(mol):
    try:
        res = Chem.Descriptors.ExactMolWt(mol)
        return res
    except:
        return np.nan
    
def HBA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBA(mol)
        return res
    except:
        return np.nan
    
def HBD(mol):
    try:
        res = Chem.rdMolDescriptors.CalcNumHBD(mol)
        return res
    except:
        return np.nan
    
def TPSA(mol):
    try:
        res = Chem.rdMolDescriptors.CalcTPSA(mol)
        return res
    except:
        return np.nan
    
def NRB(mol):
    try:
        res =  Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        return res
    except:
        return np.nan
    
def get_num_rings(mol):
    try:
        r = mol.GetRingInfo()
        res = len(r.AtomRings())
        return res
    except:
        return np.nan

def get_num_rings_6(mol):
    try:
        r = mol.GetRingInfo()
        res = len([x for x in r.AtomRings() if len(x) > 6])
        return res
    except:
        return np.nan

def LOGP(mol):
    try:
        res = logP(mol)
        return res
    except:
        return np.nan
    
def MCF(mol):
    """
    Keep molecules whose MCF=True
    MCF=True means toxicity. but toxicity=True is not bad if the patient is dying.
    """
    try:
        res = mol_passes_filters(mol)
        return res
    except:
        return np.nan

def synthesis_availability(mol):
    """
    0-10. smaller, easier to synthezie.
    not very accurate.
    """
    try:
        res = SA(mol)
        return res
    except:
        return np.nan
    
def estimation_drug_likeness(mol):
    """
    0-1. bigger is better.
    """
    try:
        res = QED(mol)
        return res
    except:
        return np.nan

def add_mol(df):
    df['ROMol'] = df['SMILES'].apply(SMILES2MOL)
    return df

def add_descriptors(df):
    df['MW'] = df.ROMol.apply(MW)
    df['logP'] = df.ROMol.apply(LOGP)
    df['HBA'] = df.ROMol.apply(HBA)
    df['HBD'] =  df.ROMol.apply(HBD)
    df['TPSA'] = df.ROMol.apply(TPSA)
    df['NRB'] = df.ROMol.apply(NRB)
    df['MCF'] = df.ROMol.apply(MCF)
    df['SA'] = df.ROMol.apply(synthesis_availability)
    df['QED'] = df.ROMol.apply(estimation_drug_likeness)
    df['rings'] = df.ROMol.apply(get_num_rings)
    return df

def add_features(df):
    df = parallelize_dataframe(df, add_descriptors)
    df = df.dropna(subset=['MW', 'logP', 'HBA', 'HBD', 'TPSA', 'NRB', 'MCF', 
                           'SA', 'QED', 'rings'])
    return df

def validity_filter(df):

    print("Start to remove invalid SMILES...")
    
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, add_mol)
        df.dropna(subset=['ROMol'], inplace=True)
    print("Finished.")
    return df

def property_filter(df, condition):
    """
    -----------------
    descriptor filter
    -----------------
    """
    print('descriptor filter')
    df = parallelize_dataframe(df, add_mol)
    df.dropna(subset=['ROMol'], inplace=True)
    df = four_rings_filter(df)
    df = add_features(df)
    df[['MW', 'logP', 'TPSA', 'SA', 'QED']] = df[['MW', 'logP', 'TPSA', 'SA',\
        'QED']].apply(lambda x: round(x, 3))
    df = df.query(condition)
    df = df.reset_index(drop=True)
    return df

def mol2ECFP4(mol, nbits=1024, radius=2):
    try:
        res = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return res
    except:
        return np.nan

def add_ECFP4(df):
    df['ECFP4'] = df['ROMol'].apply(mol2ECFP4)
    return df

def fingerprint_similarity(line): 
    tanimoto = DataStructs.FingerprintSimilarity(line[0], line[1])
    return tanimoto

def molecule_in_patent(sample_fingerprint, patent_fingerprints, l, ths):
    fp_list = [sample_fingerprint] * int(l)
    matrix = pd.DataFrame({'SMILES':fp_list, 'patent':patent_fingerprints})
    matrix['tanimoto'] = matrix.apply(fingerprint_similarity, axis=1)
    if len(matrix.query('tanimoto%s' % ths)) > 0:
        return True
    else:
        return False
    
def add_patent(df, patent_fingerprints, l, ths):
    molecule_in_patentv2 = partial(molecule_in_patent, 
                                   patent_fingerprints=patent_fingerprints, 
                                   l=l, ths=ths)
    df['patent'] = df['ECFP4'].apply(molecule_in_patentv2)
    return df

def hard_patent_filter(df, patent, ths='==1'):
    """
    -------------------------------
    Remove molcules those in patent
    -------------------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, add_mol)
        df.dropna(subset=['ROMol'], inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_mol)
        patent.dropna(subset=['ROMol'], inplace=True)
    df = parallelize_dataframe(df, add_ECFP4)
    patent = parallelize_dataframe(patent, add_ECFP4)
    patent_fingerprints = patent.ECFP4
    l = len(patent_fingerprints)
    df = parallelize_dataframe(df, add_patent, 
                               patent_fingerprints=patent_fingerprints, 
                               l=l, ths=ths)
    df = df[df['patent']==False]
    del df['patent']
    df = df.reset_index(drop=True)
    del df['ECFP4']
    return df, patent

def soft_patent_filter(df, patent, ths='>0.85'):
    """
    -----------------------------------
    Remove molcules Tc > 0.85 in patent
    -----------------------------------
    """
    df, patent = hard_patent_filter(df, patent, ths)
    return df, patent

def get_scaffold_mol(mol):
    try: 
        res = GetScaffoldForMol(mol)
        if not res == None and not res == "":
            return res
        else:
            return np.nan
    except:
        return np.nan

def add_atomic_scaffold(df):
    df['atomic_scaffold_mol'] = df.ROMol.apply(get_scaffold_mol)
    return df

def substruct_match(df):
    # Chem.MolFromSmiles
    return df.ROMol.HasSubstructMatch(df.patent_scaffold_mol)

def add_substructure_match(matrix, outname='remain'):
    matrix[outname] = matrix.apply(substruct_match, axis=1)
    return matrix

def scaffold_in_patent(mol, patent_scaffolds, l):
    mol_list = [mol] * int(l)
    matrix = pd.DataFrame({'ROMol':mol_list, 
                           'patent_scaffold_mol':list(patent_scaffolds)})
    matrix = add_substructure_match(matrix)
    if len(matrix.query('remain==True')) > 0:
        return False
    else:
        return True 
    
def judge_substructure(df, col, patent_scaffolds, l, outname='remain'):
    scaffold_in_patentv2 = partial(scaffold_in_patent, 
                     patent_scaffolds=patent_scaffolds,
                     l=l)
    df[outname] = df[col].apply(scaffold_in_patentv2)
    return df

def atom_scaffold_filter(df, patent, col = 'atomic_scaffold_mol'):
    """
    ----------------------
    atomic scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, add_mol)
        df.dropna(subset=['ROMol'], inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_mol)
        patent.dropna(subset=['ROMol'], inplace=True)
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold)
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold)
    patent_scaffolds = set(patent[col])
    l = len(patent_scaffolds)
    df = parallelize_dataframe(df, judge_substructure, 
                               col = col,
                               patent_scaffolds=patent_scaffolds, l=l)
    df = df[df['remain']==True]
    del df['remain']
    df = df.reset_index(drop=True)
    return df, patent

def get_graph_scaffold_mol(atomic_scaffold_mol):
    try:
        #atomic_scaffold_mol.Compute2DCoords()
        graph_scaffold_mol = MurckoScaffold.MakeScaffoldGeneric( 
            atomic_scaffold_mol)
        return graph_scaffold_mol
    except:
        return np.nan
    
def add_graph_scaffold(df, col='atomic_scaffold_mol', 
                       outname='graph_scaffold_mol'):
    df[outname] = df[col].apply(get_graph_scaffold_mol)
    return df

def grap_scaffold_filter(df, patent, col='graph_scaffold_mol'):
    """
    ----------------------
    graph scaffold filter
    ----------------------
    """
    if "ROMol" not in df.columns:
        df = parallelize_dataframe(df, add_mol)
        df.dropna(subset=['ROMol'], inplace=True)
    if "ROMol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_mol)
        patent.dropna(subset=['ROMol'], inplace=True)
    if "atomic_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_atomic_scaffold)
    if "atomic_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_atomic_scaffold)
    if "graph_scaffold_mol" not in df.columns:
        df = parallelize_dataframe(df, add_graph_scaffold)
    if "graph_scaffold_mol" not in patent.columns:
        patent = parallelize_dataframe(patent, add_graph_scaffold)
    df, patent = atom_scaffold_filter(df, patent, col = col)
    return df, patent

def mol2smiles(mol):
    try:
        sm = Chem.MolToSmiles(mol)
        if not sm == None and not sm == "":
            return sm
        else:
            return np.nan
    except:
        return np.nan

def atomic_scaffold(df):
    if "atomic_scaffold" not in df.columns:
        df["atomic_scaffold"] = df["atomic_scaffold_mol"].apply(mol2smiles)
    return df

def graph_scaffold(df):
    if "graph_scaffold" not in df.columns:
        df["graph_scaffold"] = df["graph_scaffold_mol"].apply(mol2smiles)
    return df

def save_file(df, path):
    if "ROMol" in df.columns:
        del df["ROMol"]
    if "atomic_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, atomic_scaffold)
        del df['atomic_scaffold_mol']
    if "graph_scaffold_mol" in df.columns:
        df = parallelize_dataframe(df, graph_scaffold)
        del df["graph_scaffold_mol"]
    df.to_csv(path, index=False)


def filter_molecule(input_file, output_dir, condition_file, patent_file):
    try:
        with open(condition_file, 'r') as f:
            condition = f.readline()
            condition = condition.strip()
    except:
        print("Read condition file failed.")
        return
    try:
        df = pd.read_csv(input_file)
    except:
        print("Read compound file failed.")
        return
    try:
        patent = pd.read_csv(patent_file)
    except:
        print("Read patent file failed.")
        return
    df = property_filter(df, condition)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_file(df, os.path.join(output_dir, "property_filter.csv"))
    df, patent = hard_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "hard_pat_filter.csv"))
    df, patent = soft_patent_filter(df, patent)
    save_file(df, os.path.join(output_dir, "soft_pat_filter.csv"))
    df, patent = atom_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "atom_pat_filter.csv"))
    df, patent = grap_scaffold_filter(df, patent)
    save_file(df, os.path.join(output_dir, "grap_pat_filter.csv"))
    return os.path.join(output_dir, "soft_pat_filter.csv")