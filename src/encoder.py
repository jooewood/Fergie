#!/usr/bin/env python3
import torch
from torch import nn
from tokenizer import encode, get_vocab_size


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False, l_token=100):
        super(RNNEncoder, self).__init__()

        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))
        self.l_token = l_token
    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        sm_list = ['Cn1[nH]c(=O)c2c(C(=O)N=c3[nH]c4ccccc4[nH]3)cc(C3CC3)nc21']
        """

        tokens, lens = encode(sm_list, self.l_token)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.rnn(self.embs(to_feed))[0]
        outputs = outputs[lens, torch.arange(len(lens))] # ???????????
        return self.final_mlp(outputs)