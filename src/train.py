#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# @author: zdx
# =============================================================================
import matplotlib
matplotlib.use("Agg")

import os
import gentrl
import encoder
import decoder
import dataloader
import torch
import time
from torch.utils.data import DataLoader
from moses.metrics.utils import get_mol
import matplotlib.pyplot as plt
from rdkit.Chem import Draw



def main(
         active_file, # Name of csv file which contains active inhibitors of 
                      # interest target.
         family_file, # Name of csv file which contains active inhibitors of
                      # the same family of interest target. 
         non_family_file, # Name of csv file which contains active inhibitors 
                          # of different family of interest target.
         molecule_file,  # Name of csv file which contains compounds selected from 
                     # ZINC database.
         output_dir,
         l_token,
         property_name,
         batch_size, 
         num_epochs,
         latent_size,
         active_ratio,
         molecule_ratio,
         family_ratio,
         non_family_ratio,
         num_workers,
         learning_rate,
         distribution_size,
         ):
    if family_file is None:
        family_ratio = 0
        non_family_ratio = 0.2
    Prop = [active_ratio, molecule_ratio, family_ratio, non_family_ratio]
    time_start=time.time()
    enc = encoder.RNNEncoder(latent_size=latent_size, l_token=l_token)
    
    dec = decoder.DilConvDecoder(latent_input_size=latent_size, 
        split_len=l_token)
    model = gentrl.GENTRL(enc, dec, latent_size * [('c', 10)], [('c', 10)], 
        latent_size=latent_size, tt_int=30, beta=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    #model.cuda()

    path_active = active_file
    path_zinc = molecule_file
    family_path = family_file
    non_family_path = non_family_file
    
    sources = []
    for path, ratio in zip(
        [path_active, path_zinc, family_path, non_family_path], 
        [active_ratio, molecule_ratio, family_ratio, non_family_ratio],      
        ):
        if ratio != 0:
            sources.append({
                'path':path,
                'smiles': 'SMILES',
                'prob': ratio,
                property_name : property_name})
    md = dataloader.MolecularDataset(sources, props=[property_name])
    train_loader = DataLoader(md, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True)
    task_path = os.path.join(output_dir, "model", '_'.join([str(Prop[0]), str(l_token)]))
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    loss = model.train_as_vaelp(train_loader, num_epochs=num_epochs, lr=learning_rate, 
        model_path=task_path, verbose_step=10, distribution_size=distribution_size)
    model.save(task_path)   #save mode
    time_end=time.time()
    time_cost = round(time_end-time_start, 4)
    generated = []
    while len(generated) < 200:
        sampled = model.sample(100, latent_size)
        sampled_valid = [s for s in sampled if get_mol(s)]
        generated += sampled_valid
    sampled_valid = generated[0:200]
    img = Draw.MolsToGridImage([get_mol(s) for s in sampled_valid], 
        molsPerRow=10, subImgSize=(300,300))
    img.save( os.path.join(task_path, 'gen_mol.jpg') )
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(loss, label='Value of Loss function')
    ax1.legend(loc='upper right')
    plt.xlabel('Training Epochs')
    plt.title('Batchsize %d   codesize %d   epoch %d\ntime cost: %d s' % ( 
        batch_size, latent_size, num_epochs, time_cost))
    plt.savefig(os.path.join(task_path, 'loss.jpg')) 
    plt.close()
    print('{} time cost'.format(task_path), time_cost, 's')

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, required=True, 
        help="Folder path saves the trained model and generated molecules.")
    ap.add_argument("-a", "--active_file", type=str, required=True,
        help="File contains inhibitors of the specific target.")
    ap.add_argument("-f", "--family_file", type=str, default=None, 
        help="File contains inhibitors of the same family of the specific target.")
    ap.add_argument("-n", "--non_family_file", type=str, required=True, 
        help="File contains inhibitors of different family of the specific target.")
    ap.add_argument("-m", "--molecule_file", type=str, required=True,
        help="FIle contains small molecules have smiliar properties to the"
             "inhibitors of the specific target.")
    ap.add_argument("-t", "--token_length", type=int, default=100,
        help="Set the token length for the model to learn.")
    ap.add_argument("-d", "--distribution_size", type=int, default=5000,
        help="Control the size of distribution.")
    ap.add_argument("-p", "--property_name", type=str, default="plogP")
    ap.add_argument("-b", "--batch_size", type=int, default=1024)
    ap.add_argument("-e", "--num_epochs", type=int, default=10)
    ap.add_argument("-l", "--latent_size", type=int, default=50)
    ap.add_argument("--active_ratio", type=float, default=0.5)
    ap.add_argument("--molecule_ratio", type=float, default=0.3)
    ap.add_argument("--family_ratio", type=float, default=0.1)
    ap.add_argument("--non_family_ratio", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    
    ap.add_argument('-g', '--gpuUSE', action='store_true', default=False,
        help='gpu use or not')
    ap.add_argument('--gpu', type=int, default=0, 
        choices=list(range(torch.cuda.device_count())),help='Which GPU to use') 
                                                                  #previous default is 0
    #ap.add_argument("-g", "--gpu_id", type=int, default=-1, 
    #    help="Tell the program which GPU it can use.")             #previous default is 0 
    ap.add_argument("-c", "--num_workers", type=int, default=24, 
        help="Set the number of cpu to load data during the training.")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)                              #previous variable is gpu_id
    main(
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