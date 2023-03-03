#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import torch
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rdkit.Chem import AllChem as Chem
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


def get_fp(list_of_smi):
    """ Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    mols = [Chem.MolFromSmiles(x) for x in list_of_smi]
    # if rdkit can't compute the fingerprint on a SMILES
    # we remove that SMILES
    idx_to_remove = []
    for idx,mol in enumerate(mols):
        try:
            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 2, 
                                                        useFeatures=False)
            fingerprints.append(fprint)
        except:
            idx_to_remove.append(idx)
    
    smi_to_keep = [smi for i,smi in enumerate(list_of_smi) if i not in\
        idx_to_remove]
    return fingerprints, smi_to_keep
def get_embedding(data):
    """ Function to compute the UMAP embedding"""            
    data_scaled = StandardScaler().fit_transform(data)
    
    embedding = umap.UMAP(n_neighbors=10,
                          min_dist=0.5,
                          metric='correlation',
                          random_state=16).fit_transform(data_scaled)
    
    return embedding

def draw_picture(file_embedding, file_name,output_file, figure_length, 
                 figure_width, labelsize = 60):
    print("draw picture")
    file_len = len(file_embedding)
    #fig, ax = plt.subplots(figsize=(60, 40))  #set the size of figure
    fig, ax = plt.subplots(figsize=(figure_length, figure_width))

    contour_c='#444444'
    colors = ['#1575A4','#D55E00','#483D8B', '#A0E0FF',  '#FFAE6E', '#FF6347']
    plt.xlim([np.min(file_embedding[0][:,0])-0.5, 
              np.max(file_embedding[0][:,0])+1.5])
    plt.ylim([np.min(file_embedding[0][:,1])-0.5, 
              np.max(file_embedding[0][:,1])+0.5])
    
    plt.xlabel('UMAP 1', fontsize=labelsize)
    plt.ylabel('UMAP 2', fontsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i in range(file_len):
        plt.scatter(file_embedding[i][:, 0], file_embedding[i][:, 1], lw=0, 
                    c=colors[i], label=file_name[i], alpha=1.0, s=(100+80*i), 
                    marker="o", 
                   edgecolors=contour_c, linewidth=1 ) 
    leg = plt.legend(prop={'size': labelsize}, loc='upper right', 
                     markerscale=3.00)
    leg.get_frame().set_alpha(0.9)    
    plt.setp(ax, xticks=[], yticks=[])
    plt.autoscale(tight=True) 
    plt.savefig(output_file, dpi=600, format='eps')

def main(input_files, output_file, figure_length, figure_width): 
    file_embedding = []
    for current_file in input_files:
        file_data = pd.read_csv(current_file)
        print("Read {} complete".format(current_file))
        print("extract SMILES.....")
        current_smiles = file_data["SMILES"]
        print("finish")
        print("get fingerprint from SMILES....")
        current_fp, current_sm = get_fp(current_smiles)
        current_fp = np.array(current_fp)
        print("finish")
        
        print("strat embedding...")
        embedding_cur = get_embedding(current_fp)
        file_embedding.append(embedding_cur)
    file_name = [os.path.splitext(os.path.split(f)[-1])[0] for f in input_files]
    draw_picture(file_embedding, file_name, output_file, figure_length, figure_width)
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_files", nargs='+',help="Write according to the file size, up to 6")
    ap.add_argument("-o", "--output_file", help="where to svae the picture")
    ap.add_argument("-l", "--figure_length", type=int, default=100)
    ap.add_argument("-w", "--figure_width", type=int, default=80)
    args = ap.parse_args()
    main(args.input_files, args.output_file, args.figure_length, args.figure_width)