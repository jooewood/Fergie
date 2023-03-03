#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
import os
from utils import validity_filter, add_features
import matplotlib.pyplot as plt

def summary_molecules(df, output_dir):
    columns = ['MW', 'logP', 'HBA', 'HBD', 'TPSA', 'NRB', 'SA', 'QED', 'rings']
    df = df[columns]

    df_summary = df.describe([.05, .5, .95])
    df_summary = df_summary.apply(lambda x: round(x, 2))
    df_path = os.path.join(output_dir, 'statistic.csv')
    df_summary.to_csv(df_path)
    
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    df.hist(layout=(5,2), figsize=(15,12), bins=50, ax=ax)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    fig_path = os.path.join(output_dir, 'statistic.png')
    fig.savefig(fig_path)


def main(inputfile, outputdir):
    # remove invalid molecules
    df = pd.read_csv(inputfile)
    df = validity_filter(df)
    # add features
    df = add_features(df)
    # draw statistics picture and output statistics table
    summary_molecules(df, outputdir)

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str, required=True,
        help="Path of input file.")
    ap.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Path of output folder.")
    args = ap.parse_args()
    main(args.input_file, args.output_dir)
    
        
    