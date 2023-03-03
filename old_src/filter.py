#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
from moses.metrics import QED, SA, logP
from rdkit.Chem import PandasTools
from rdkit import Chem
from moses.metrics.utils import get_n_rings, get_mol
from parallelize_apply import parallelize_dataframe
from moses.metrics import mol_passes_filters, QED, SA, logP
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
params = {'axes.titlesize':'18',
          'xtick.labelsize':'18',
          'ytick.labelsize':'18'}
matplotlib.rcParams.update(params)

def filiter_mol(smiles):
    mol = get_mol(smiles)
    if not mol == None:
        return mol
    else:
        return 'NA'
        
def mol_pass(data):
    data['ROMol'] = data['SMILES'].apply(filiter_mol)
    return data

def get_num_rings(mol):
    r = mol.GetRingInfo()
    return len(r.AtomRings())

def add_properties(data):
    data['MW'] = data.ROMol.apply(Chem.Descriptors.ExactMolWt)
    data['logP'] = data.ROMol.apply(logP)
    data['HBA'] = data.ROMol.apply(Chem.rdMolDescriptors.CalcNumHBA)
    data['HBD'] =  data.ROMol.apply(Chem.rdMolDescriptors.CalcNumHBD)
    data['TPSA'] = data.ROMol.apply(Chem.rdMolDescriptors.CalcTPSA)
    data['NRB'] = data.ROMol.apply(Chem.rdMolDescriptors.CalcNumRotatableBonds)
    data['MCF'] = data.ROMol.apply(mol_passes_filters)
    data['SA'] = data.ROMol.apply(SA)
    data['QED'] = data.ROMol.apply(QED)
    data['rings'] = data.ROMol.apply(get_num_rings)
    return data

def filter_mol(df):
    df = parallelize_dataframe(df, mol_pass)
    df = df.query('ROMol!=NA')
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = parallelize_dataframe(df, add_properties)
    return df

## Statistics of positive

columns = ['MW', 'logP', 'HBA', 'HBD', 'TPSA', 'NRB', 'SA', 'QED', 'rings']

def add_descriptors(path):
    df = pd.read_csv(path)
    df = df[['ID', 'SMILES']]
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = add_properties(df)
    df = df[columns]
    return df

def reshape_df(raw, type_):
    for i, col in enumerate(columns):
        #  i = 0
        if i==0:
            df = pd.DataFrame({
                "value": raw[col].values,
                "descriptor":[col] * len(raw)
                })
        else:
            tmp = pd.DataFrame({
                "value": raw[col].values,
                "descriptor":[col] * len(raw)
                })
            df = pd.concat([df,tmp], sort=False)
    df['data'] = type_
    return df
df1_name = 
df2_name = 
path1 = '/home/tensorflow/SuperMolDesign/data/PPK1/PPK1_5.csv'
path2 = '/home/tensorflow/SuperMolDesign/Molecule_generated/PPK1_5_MW500/0_four_rings_filter.csv'
active = add_descriptors(path1)
active_re = reshape_df(active, "Patent")
design = add_descriptors(path2)
design_copy = design.copy()
#design = design[design["HBD"]>1]
design_re = reshape_df(design, "AI Design")
df = pd.concat([active_re, design_re], sort=False)

fig = plt.figure(figsize=(20, 8), dpi=600)
axes = fig.subplots(1,len(columns))
for i,col in enumerate(columns):
    tmp = df.query('descriptor=="%s"' % col)
    sns.violinplot(x="descriptor",y="value", data=tmp,hue="data", split=True, palette="Set3", ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].legend('')
fig.tight_layout()
out = '/home/tensorflow/Desktop/work/CloudStation/now/distribution'
name = '_'.join(df1_name, df2_name)
out = '/'.join([out, df1_name, df2_name]) + '.eps'
fig.savefig(out,dpi=600,format='eps')

def summary_mole(path):
    df = pd.read_csv(path)
    df = df[['ID', 'SMILES']]
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
    df = add_properties(df)
    df = df[columns]
    pre = os.path.dirname(path)
    fig_name = path.split('/')[-1]
    fig_name = fig_name.split('.csv')[0]
    df_summary = df.describe([.05, .5, .95])
    df_summary = df_summary.apply(lambda x: round(x, 2))
    df_summary.to_csv('%s/%s_statistics.csv' % (pre, fig_name))
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    df.hist(layout=(5,2), figsize=(15,12), bins=50, ax=ax)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    fig.savefig('%s/%s_statistics.png' % (pre, fig_name))
summary_mole(path1)
summary_mole(path2)

# =================== draw lenth plot ========================================
path = '/home/tensorflow/SuperMolDesign/data/MW500_PPK1/PPK1_6.csv'
def draw_hist(path, Bins=20):
    df = pd.read_csv(path)
    def get_len(x):
        return len(x)
    df['len'] = df['SMILES'].apply(get_len)
    x = df['len']#.describe()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(x,bins=Bins,color='blue')
    plt.show()
draw_hist(path1, 10)

# MDfilter
PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
df_pro = add_properties(df)
del df['ROMol']
df = df_pro.query('300<=MW & MW<=600 & -2<=logP & logP<=6 & HBD<=5 & HBA<=10 & TPSA<150 & NRB<10 & MCF==True')
df = df[0:494]
df.to_csv('../data/MD2_library_filtered.csv', index=False)
summary = df_pro.describe()
# df_box.boxplot(layout=(2,5), figsize=(15,12), by='property')

