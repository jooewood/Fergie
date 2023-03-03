#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from math import pi, log
from lp import LP
import pickle
import pandas as pd
from moses.metrics.utils import get_mol
import os
import numpy as np
from rdkit.Chem import Draw
import torchsnooper
from PIL import Image, ImageChops
import matplotlib.pyplot as plt

from rdkit.Chem import DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem
from tqdm import tqdm

class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def print(self):
        for key in self.stats.keys():
            print(str(key) + ": {:4.4};".format(
                sum(self.stats[key]) / len(self.stats[key])
            ), end='')

        print()

class GENTRL(nn.Module):
    '''
    GENTRL model
    '''
    def __init__(self, enc, dec, latent_descr, feature_descr, tt_int=40,
                 tt_type='usual', latent_size=50, beta=0.01, gamma=0.1):
        super(GENTRL, self).__init__()
        self.latent_size = latent_size
        self.enc = enc
        self.dec = dec

        self.num_latent = len(latent_descr) # latent_descr = 50 * [('c', 20)]
        self.num_features = len(feature_descr) # feature_descr = [('c', 20)]

        self.latent_descr = latent_descr
        self.feature_descr = feature_descr

        self.tt_int = tt_int
        self.tt_type = tt_type

        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type)

        self.beta = beta
        self.gamma = gamma
        
    #@torchsnooper.snoop()
    def get_elbo(self, x, y):
        means, log_stds = torch.split(self.enc.encode(x),
                                      len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = self.dec.weighted_forward(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        zy = torch.cat([latvar_samples, y], dim=1)
        log_p_zy = self.lp.log_prob(zy)

        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)
        log_p_z_by_y = log_p_zy - log_p_y
        log_p_y_by_z = log_p_zy - log_p_z

        kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()

        elbo = rec_part - self.beta * kldiv_part
        elbo = elbo + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy(),
            'log_p_y_by_z': log_p_y_by_z.mean().detach().cpu().numpy(),
            'log_p_z_by_y': log_p_z_by_y.mean().detach().cpu().numpy()
        }

    def save(self, folder_to_save='./'):
        if folder_to_save[-1] != '/':
            folder_to_save = folder_to_save + '/'
        torch.save(self.enc.state_dict(), folder_to_save + 'enc.model')
        torch.save(self.dec.state_dict(), folder_to_save + 'dec.model')
        torch.save(self.lp.state_dict(), folder_to_save + 'lp.model')

        pickle.dump(self.lp.order, open(folder_to_save + 'order.pkl', 'wb'))

    def load(self, folder_to_load='./'):
        if folder_to_load[-1] != '/':
            folder_to_load = folder_to_load + '/'

        order = pickle.load(open(folder_to_load + 'order.pkl', 'rb'))
        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type,
                     order=order)

        self.enc.load_state_dict(torch.load(folder_to_load + 'enc.model'))
        self.dec.load_state_dict(torch.load(folder_to_load + 'dec.model'))
        self.lp.load_state_dict(torch.load(folder_to_load + 'lp.model'))
    
    #@torchsnooper.snoop()
    def train_as_vaelp(self, train_loader, num_epochs=10,
                       verbose_step=50, distribution_size=5000 ,lr=1e-3, model_path='/home/tensorflow/Desktop'):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        global_stats = TrainStats()
        local_stats = TrainStats()

        epoch_i = 0
        to_reinit = False
        buf = None

        loss_list = []
        #label = 0
        count = 0
        while epoch_i < num_epochs:
            i = 0
            if verbose_step:
                print("Training for Epoch", epoch_i, ":")

            if epoch_i in [0, 1, 5]:
                to_reinit = True

            epoch_i += 1
            for x_batch, y_batch in tqdm(train_loader):
                #try:
#                if verbose_step:
#                    print("!", end='')
                i += 1
                count += 1
                y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.view(-1, 1).contiguous()

                if to_reinit:
                    if (buf is None) or (buf.shape[0] < distribution_size): 
                        enc_out = self.enc.encode(x_batch)
                        means, log_stds = torch.split(enc_out,
                                                      len(self.latent_descr),
                                                      dim=1)
                        z_batch = (means + torch.randn_like(log_stds) *
                                   torch.exp(0.5 * log_stds))
                        cur_batch = torch.cat([z_batch, y_batch], dim=1)
                        if buf is None:
                            buf = cur_batch
                        else:
                            buf = torch.cat([buf, cur_batch])
                    else:
                        descr = len(self.latent_descr) * [0]
                        descr += len(self.feature_descr) * [1]
                        self.lp.reinit_from_data(buf, descr)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        self.lp.to(device)
                        #self.lp.cuda()
                        buf = None
                        to_reinit = False

                    continue

                elbo, cur_stats = self.get_elbo(x_batch, y_batch)
                local_stats.update(cur_stats)
                global_stats.update(cur_stats)

                optimizer.zero_grad()
                loss = -elbo
                loss.backward()
                optimizer.step()
                ## Draw picture to generate movie
#                if num_epochs > 3:
#                    generated = []
#                    while len(generated) < 16:
#                        z = self.lp.sample(200, self.latent_size * ['s'] + ['m'])
#                        sampled = self.dec.sample(self.latent_size, z, argmax=False)
#                        sampled_valid = [s for s in sampled if get_mol(s)]
#                        generated += sampled_valid
#                    sampled_valid = generated[0:16]                
#                    mol_img = Draw.MolsToGridImage([get_mol(s) for s in sampled_valid], molsPerRow=4, subImgSize=(300,300))
#                    tmp_folder = '{}/picture_title'.format(model_path)
#                    if not os.path.exists(tmp_folder):
#                        os.system('mkdir -p {}'.format(tmp_folder))
#                    img_path = '{}/iteration{}.jpg'.format(tmp_folder, count)
#                    mol_img.save(img_path, 'JPEG', quality = 95, subsampling=0)

#                    plt.imshow(img)
#                    if label != 0:
#                        plt.close()
#                    fig = plt.figure()   
#                    fig.set_size_inches(15, 15)
#                    fig.canvas.manager.window.move(300,300)
#                    plt.imshow(img)
#                    plt.axis('off')
#                    plt.gca().xaxis.set_major_locator(plt.NullLocator())    
#                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
#                    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
#                    plt.margins(0,0)
#                    plt.pause(0.5)
#                    label = 1
                
                if verbose_step and i % verbose_step == 0:
                    #local_stats.print()
                    local_stats.reset()
                    i = 0
            loss_list.append(loss.cpu().detach().numpy())
            #epoch_i += 1
            if i > 0:
                #local_stats.print()
                local_stats.reset()
        return loss_list
        #return global_stats
    
    def train_as_rl(self,
                    reward_fn, path,
                    num_iterations=50000, verbose_step=50,
                    batch_size=200,
                    cond_lb=-2, cond_rb=0,
                    lr_lp=1e-5, lr_dec=1e-6):
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)

#        global_stats = TrainStats()
#        local_stats = TrainStats()

        cur_iteration = 0
        iter_list = []
        mean_reward = []
#        valid_perc = []
        pdar = tqdm(total=num_iterations)
        while cur_iteration < num_iterations:
            #print(cur_iteration)
            
            expl_proportion = 0.1
            exploit_size = int(batch_size * (1 - expl_proportion))

            exploit_z = self.lp.sample(exploit_size, self.latent_size * ['s'] + ['m'])
            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            expl_size = int(batch_size * expl_proportion)
            expl_z = torch.randn(expl_size, exploit_z.shape[1])
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            z = torch.cat([exploit_z, expl_z])
            smiles = self.dec.sample(self.latent_size, z, argmax=False)
            
            zc = torch.zeros(z.shape[0], 1).to(z.device)
            conc_zy = torch.cat([z, zc], dim=1)
            log_probs = self.lp.log_prob(conc_zy, marg=self.latent_size * [False] + [True])
            log_probs += self.dec.weighted_forward(smiles, z)
            
            r_list = list(map(reward_fn, smiles))
            tmp_mean_reward = np.mean(r_list)
            iter_list.append(cur_iteration)
            mean_reward.append(round(tmp_mean_reward,4))
            rewards = torch.tensor(r_list).float().to(exploit_z.device)
            rewards_bl = rewards - rewards.mean()

            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            cur_iteration += 1

            if verbose_step and (cur_iteration + 1) % verbose_step == 0:
                tmp_mean_reward = np.mean(r_list)
                iter_list.append(cur_iteration)
                mean_reward.append(round(tmp_mean_reward,4))
            pdar.update(1)
      
        df = pd.DataFrame({'mean_reward':mean_reward, 'iteration':iter_list})
        print(df)
        df.to_csv(path, index=False)
        return df
        #return global_stats

    def sample(self, num_samples, latent_size=50):
        z = self.lp.sample(num_samples, latent_size * ['s'] + ['m'])
        smiles = self.dec.sample(latent_size, z, argmax=False)
        return smiles
