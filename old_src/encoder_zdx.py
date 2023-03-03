#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import torch
from torch import nn
from tokenizer import encode, get_vocab_size
hidden_size=256
num_layers=2
latent_size=50
bidirectional=False

class RNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        super(RNNEncoder, self).__init__()

        embs = nn.Embedding(get_vocab_size(), hidden_size)
        rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        sm_list = ['Cn1[nH]c(=O)c2c(C(=O)N=c3[nH]c4ccccc4[nH]3)cc(C3CC3)nc21']
        """

        tokens, lens = encode(sm_list) # tokens[2, 100]
        to_feed = tokens.transpose(1, 0).to(embs.weight.device) # to_feed[100, 2]
        emb_matrix = embs(to_feed) # emb_matrix[100, 2, 256]
        outputs = rnn(emb_matrix)[0] # outputs[100, 2, 256]
        outputs_1 = outputs[lens, torch.arange(len(lens))] # outputs_1[2, 256]
        enc_out = final_mlp(outputs_1) 
        return enc_out # [2, 100]

means, log_stds = torch.split(enc_out, 50, dim=1) # means[2,50] log_stds[2,50]
z_batch = (means + torch.randn_like(log_stds) * torch.exp(0.5 * log_stds)) # z_batch[2,50]
cur_batch = torch.cat([z_batch, y_batch], dim=1)
