#!/usr/bin/env python3

import glob
import argparse
import pandas as pd
from os.path import isfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_path', type=str, required=True,
                        help='Path to get original dataset files')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save merged dataset file')
    return parser


def main(config):
    dir_path = config.origin_path
    onlyfiles = [f for f in glob.glob(dir_path) if isfile(f)]

    df_list = []
    for file in onlyfiles:
        df_list.append(pd.read_csv(file, sep=";"))
    df_big = pd.concat(df_list)
    df_big.to_csv(config.save_path)    


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)