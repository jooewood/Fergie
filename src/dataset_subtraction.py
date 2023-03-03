#!/usr/bin/env python3

import argparse
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--big_dataset', type=str, required=True,
                        help='Path to get subtracted dataset')
    parser.add_argument('--small_dataset', type=str, required=True,
                        help='Path to get dataset need to be reduced')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save processed dataset file')
    return parser


def main(config):
    df_big = pd.read_csv(config.big_dataset)
    df_small = pd.read_csv(config.small_dataset)
    IDs = df_small['ID'].values
    df_diff = df_big.loc[~df_big['ID'].isin(IDs)]
    df_diff.to_csv(config.save_path)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)