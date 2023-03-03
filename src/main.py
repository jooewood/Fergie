#!/usr/bin/env python3

from preprocess import preprocess
import torch
from train import main
from initial_sampling import init_sample
from argparse import ArgumentParser
from utils import filter_molecule

if __name__ == '__main__':
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="subparser_name")
    
    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("-f", "--preprocess_file", type=str, required=True, 
        nargs="+", help="Files which will be preprocessed.")
    preprocess_parser.add_argument("-o", "--output_dir", type=str, default=None,
        help="Folder path to save the preprocessed molecules.")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Folder path saves the trained model and generated molecules.")
    train_parser.add_argument("-a", "--active_file", type=str, required=True,
        help="File contains inhibitors of the specific target.")
    train_parser.add_argument("-f", "--family_file", type=str, default=None,
        help="File contains inhibitors of the same family of the specific target.")
    train_parser.add_argument("-n", "--non_family_file", type=str, required=True, 
        help="File contains inhibitors of different family of the specific target.")
    train_parser.add_argument("-m", "--molecule_file", type=str, required=True,
        help="FIle contains small molecules have smiliar properties to the"
             "inhibitors of the specific target.")
    train_parser.add_argument("-t", "--token_length", type=int, default=100,
        help="Set the token length for the model to learn.")
    train_parser.add_argument("-d", "--distribution_size", type=int, default=5000,
        help="Control the size of distribution.")
    train_parser.add_argument("-p", "--property_name", type=str, default="plogP")
    train_parser.add_argument("-b", "--batch_size", type=int, default=1024)
    train_parser.add_argument("-e", "--num_epochs", type=int, default=10)
    train_parser.add_argument("-l", "--latent_size", type=int, default=50)
    train_parser.add_argument("--active_ratio", type=float, default=0.5)
    train_parser.add_argument("--molecule_ratio", type=float, default=0.3)
    train_parser.add_argument("--family_ratio", type=float, default=0.1)
    train_parser.add_argument("--non_family_ratio", type=float, default=0.1)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument('-g', '--gpuUSE', action='store_true', default=False,
        help='gpu use or not')
    train_parser.add_argument('--gpu', type=int, default=0, 
        choices=list(range(torch.cuda.device_count())),help='Which GPU to use') 
    train_parser.add_argument("-c", "--num_workers", type=int, default=24, 
        help="Set the number of cpu to load data during the training.")
    
    initial_sampling_parser = subparsers.add_parser("initial_sampling")
    initial_sampling_parser.add_argument("-i", "--input_dir", type=str, required=True, 
        help="Location of model saved.")
    initial_sampling_parser.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Location to save the generated molecule.")
    initial_sampling_parser.add_argument("-p", "--protein", type=str, required=True, 
        help="The name of the target.")
    initial_sampling_parser.add_argument("-l", "--latent_size", type=int, default=50,
        help="Dimension of latent code size.")
    initial_sampling_parser.add_argument("-n", "--num_sample", type=int, default=30000,
        help="Total number of molecule to sample.")
    initial_sampling_parser.add_argument("-b", "--batch_size", type=int, default=2000, 
        help="Number of molecule to sample in a batch.")
    initial_sampling_parser.add_argument("-t", "--tt_int", type=int, default=30)
    initial_sampling_parser.add_argument("--beta", type=float, default=0.001)
    initial_sampling_parser.add_argument('-g', '--gpuUSE', action='store_true', 
        default=False, help='gpu use or not')
    initial_sampling_parser.add_argument('--gpu', type=int, default=-1, 
        choices=list(range(torch.cuda.device_count())),help='Which GPU to use')
    
    total_parser = subparsers.add_parser("all")
    total_parser.add_argument("-p_o", "--preprocess_output_dir", type=str, default='/home/zdx/data/preprocessed',
        help="Folder path to save the preprocessed molecules.")
    total_parser.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Folder path saves the trained model and generated molecules.")
    total_parser.add_argument("-a", "--active_file", type=str, required=True,
        help="File contains inhibitors of the specific target.")
    total_parser.add_argument("-f", "--family_file", type=str, default=None,
        help="File contains inhibitors of the same family of the specific target.")
    total_parser.add_argument("-n", "--non_family_file", type=str, 
        default='/home/zdx/data/preprocessed/chembl_rd_filters_features_filtered_preprocess.csv',
        help="File contains inhibitors of different family of the specific target.")
    total_parser.add_argument("-m", "--molecule_file", type=str, 
        default='/home/zdx/data/preprocessed/moses_zinc_preprocess.csv',
        help="FIle contains small molecules have smiliar properties to the"
             "inhibitors of the specific target.")
    total_parser.add_argument("-t", "--token_length", type=int, default=100,
        help="Set the token length for the model to learn.")
    total_parser.add_argument("-d", "--distribution_size", type=int, default=5000,
        help="Control the size of distribution.")
    total_parser.add_argument("-c", "--condition_file", default='../condition/RO5.txt', 
        help="The file contains the property rules to filter compounds.")
    total_parser.add_argument("-p", "--property_name", type=str, default="plogP")
    total_parser.add_argument("--train_batch_size", type=int, default=1024)
    total_parser.add_argument("-e", "--num_epochs", type=int, default=10)
    total_parser.add_argument("--train_latent_size", type=int, default=50)
    total_parser.add_argument("--active_ratio", type=float, default=0.5)
    total_parser.add_argument("--molecule_ratio", type=float, default=0.3)
    total_parser.add_argument("--family_ratio", type=float, default=0.1)
    total_parser.add_argument("--non_family_ratio", type=float, default=0.1)
    total_parser.add_argument("--learning_rate", type=float, default=1e-4)
    total_parser.add_argument('-g', '--gpuUSE', action='store_true', default=False,
        help='gpu use or not')
    total_parser.add_argument('--gpu', type=int, default=0, 
        choices=list(range(torch.cuda.device_count())),help='Which GPU to use') 
    total_parser.add_argument("--num_workers", type=int, default=24, 
        help="Set the number of cpu to load data during the training.")
    total_parser.add_argument("--protein", type=str, required=True, 
        help="The name of the target.")
    total_parser.add_argument("--sampling_latent_size", type=int, default=50,
        help="Dimension of latent code size.")
    total_parser.add_argument("--num_sample", type=int, default=30000,
        help="Total number of molecule to sample.")
    total_parser.add_argument("--sampling_batch_size", type=int, default=2000, 
        help="Number of molecule to sample in a batch.")
    total_parser.add_argument("--tt_int", type=int, default=30)
    total_parser.add_argument("--beta", type=float, default=0.001)

    
    args = parser.parse_args()
    if args.subparser_name == "preprocess":
        for preprocess_file in args.preprocess_file:
            preprocess(
                input_dir = args.input_dir,
                preprocess_file = preprocess_file, 
                output_dir = args.output_dir
            )
    elif args.subparser_name == "train":
        torch.cuda.set_device(args.gpu)
        main(
            input_dir = args.input_dir,
            output_dir = args.output_dir,
            active_file = args.active_file,
            family_file = args.family_file,
            non_family_file = args.non_family_file,
            molecule_file = args.molecule_file,
            l_token = args.token_length,
            property_name = args.property_name,
            batch_size = args.batch_size,
            num_epochs = args.num_epochs,
            latent_size = args.latent_size,
            active_ratio = args.active_ratio,
            molecule_ratio = args.molecule_ratio,
            family_ratio = args.family_ratio,
            non_family_ratio = args.non_family_ratio,
            num_workers = args.num_workers,
            learning_rate = args.learning_rate,
            distribution_size = args.distribution_size
        )
    elif args.subparser_name == "initial_sampling":
        torch.cuda.set_device(args.gpu)
        init_sample(
            input_dir = args.input_dir,
            output_dir = args.output_dir,
            latent_size = args.latent_size,
            num_sample = args.num_sample,
            batch_size = args.batch_size,
            target_name = args.protein,
            tt_int = args.tt_int,
            beta = args.beta
        )
    elif args.subparser_name == "all":
        active_file = args.active_file
        family_file = args.family_file
        non_family_file = args.non_family_file
        molecule_file = args.molecule_file

        active_file = preprocess(active_file, out_dir=args.preprocess_output_dir)
        if family_file is not None:
            family_file = preprocess(family_file, out_dir=args.preprocess_output_dir)
        non_family_file = preprocess(non_family_file, out_dir=args.preprocess_output_dir)
        molecule_file = preprocess(molecule_file, out_dir=args.preprocess_output_dir)

        torch.cuda.set_device(args.gpu)
        main(
            output_dir = args.output_dir,
            active_file = active_file,
            family_file = family_file,
            non_family_file = non_family_file,
            molecule_file = molecule_file,
            l_token = args.token_length,
            property_name = args.property_name,
            batch_size = args.train_batch_size,
            num_epochs = args.num_epochs,
            latent_size = args.train_latent_size,
            active_ratio = args.active_ratio,
            molecule_ratio = args.molecule_ratio,
            family_ratio = args.family_ratio,
            non_family_ratio = args.non_family_ratio,
            num_workers = args.num_workers,
            learning_rate = args.learning_rate,
            distribution_size = args.distribution_size
        )
        init_generated_molecules_file = init_sample(
            input_dir = args.output_dir + "/model/" + str(args.active_ratio) + "_" + str(args.token_length),
            output_dir = args.output_dir,
            latent_size = args.sampling_latent_size,
            num_sample = args.num_sample,
            batch_size = args.sampling_batch_size,
            target_name = args.protein,
            tt_int = args.tt_int,
            beta = args.beta
        )
        soft_pat_filter_file = filter_molecule(
            input_file = init_generated_molecules_file,
            output_dir = args.output_dir,
            condition_file = args.condition_file,
            patent_file = active_file
        )
        train_predict_pycaret(active_file=active_file, decoy_file=non_family_file, 
                              test_file=soft_pat_filter_file, out_dir=os.path.join(args.output_dir,'classifier'))
        

