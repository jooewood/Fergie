#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
import seaborn as sns
from utils import validity_filter, add_features
import matplotlib.pyplot as plt

def reshape_df(raw, columns, type_):
    for i, col in enumerate(columns):
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


def draw_properties_comparison(left_path, right_path, output_file, title):
    df_left = pd.read_csv(left_path)
    df_right = pd.read_csv(right_path)
    df_left = validity_filter(df_left)
    df_right = validity_filter(df_right)
    df_left = add_features(df_left)
    df_right = add_features(df_right)
    columns = ['MW', 'logP', 'HBA', 'HBD', 'TPSA', 'NRB', 'SA', 'QED', 'rings']
    df_right = df_right[columns]
    df_left = df_left[columns]
    
    df_left_re = reshape_df(df_left,columns, "Real")
    df_right_re = reshape_df(df_right, columns,"AI Design")

    df = pd.concat([df_left_re, df_right_re], sort=False)

    fig = plt.figure(figsize=(20, 8), dpi=600)
    axes = fig.subplots(1, len(columns))
    for i,col in enumerate(columns):
        tmp = df.query('descriptor=="%s"' % col)
        sns.violinplot(x="descriptor", y="value", data=tmp, hue="data", 
                       split=True, palette="Set3", ax=axes[i])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].legend('')
    plt.title(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=600, format='eps')


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-l", "--left_file", required=True,
        help="Path of real molecules file show on the left side of figure.")
    ap.add_argument("-r", "--right_file", required=True,
        help="Path of AI design molecules file show on the right side of figure.")
    ap.add_argument("-t", "--title",required=True,
                    help="Picture title")
    ap.add_argument("-o", "--output_file", required=True, 
        help="Path of output folder.")
    args = ap.parse_args()
    draw_properties_comparison(args.left_file, args.right_file, args.title, args.output_file)
