#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

from utils import filter_molecule

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str, required=True, 
        help="File contains compounds will be filtered.")
    ap.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Location to save the filtered molecules.")
    ap.add_argument("-p", "--patent_file", required=True,
        help="File contains active compounds for train vae model.")
    ap.add_argument("-c", "--condition_file", default='../condition/RO5.txt', 
        help="The file contains the property rules to filter compounds.")
    args = ap.parse_args()
    filter_molecule(
        input_file = args.input_file,
        output_dir = args.output_dir,
        condition_file = args.condition_file,
        patent_file = args.patent_file
    )
