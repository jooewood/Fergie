#!/usr/bin/env python3
import torch
import re

#_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
#          'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
#          'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
#          'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
#          'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
#          'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
#          'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
#          'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
#          'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

_atoms = ['Cl', 'Br', 'Na', 'Si', 'Li', 'Se']

def get_tokenizer_re(atoms):
    return re.compile('('+'|'.join(atoms)+r'|\%\d\d|.)')

_atoms_re = get_tokenizer_re(_atoms)

#__i2t = {
#    0: 'unused', 1: '>', 2: '<', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
#    7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
#    13: '=', 14: '3', 15: ')', 16: '4', 17: '-', 18: 'n',
#    19: 'o', 20: '5', 21: 'H', 22: '(', 23: 'C',
#    24: '1', 25: 'S', 26: 's', 27: 'Br'
#}
#
#
#__t2i = {
#    '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
#    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
#    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
#    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27
#}

__i2t = {
    0: 'unused', 1: '>', 2: '<', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
    7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
    13: '=', 14: '3',  15: ')', 16: '4', 17: '-', 18: 'n',
    19: 'o', 20: '5',  21: 'H', 22: '(', 23: 'C',
    24: '1', 25: 'S',  26: 's', 27: 'Br', 28: '/',  
    29: '7', 30: '8',  31: 'I', 32: 'P', 33: '9', 34: '@', 35: 'p',
    36: '+', 37: '\\', 38: 'K', 39: 'B', 40: 'Na', 41:'Si', 42:'Li',
    43: 'Se'
}

__t2i = {
    '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
    'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
    '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
    'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27, '/': 28, 
    '7': 29, '8': 30, 'I': 31, 'P': 32, '9': 33, '@': 34, 'p': 35,
    '+': 36, '\\':37, 'K': 38, 'B': 39, 'Na': 40,'Si':41, 'Li': 42,
    'Se':43
}

def smiles_tokenizer(line, atoms=None):
    """
    Tokenizes SMILES string atom-wise using regular expressions. While this
    method is fast, it may lead to some mistakes: Sn may be considered as Tin
    or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
    specify a set of two-letter atoms explicitly.

    Parameters:
         atoms: set of two-letter atoms for tokenization
    line = s
    """
    try:
        if atoms is not None:
            reg = get_tokenizer_re(atoms)
        else:
            reg = _atoms_re
        return reg.split(line)[1::2]
    except:
        reg = "None"
        return reg

def encode(sm_list, pad_size=50):
    """
    Encoder list of smiles to tensor of tokens
    sm_list = ['C1=C(C=NN2)C2=CC(Br)=C1']
    i = 0
    s = sm_list[0]
    """
    res = []
    lens = []
    for i, s in enumerate(sm_list):
        tokens = ([1] + [__t2i[tok]
                  for tok in smiles_tokenizer(s)])[:pad_size - 1]
    #tokens is a list
        lens.append(len(tokens))
        tokens += (pad_size - len(tokens)) * [2]
    #the rest is filled with 2
        res.append(tokens)
    return torch.tensor(res).long(), lens

def get_token_zdx(s, pad_size=50):
    try:
        tokens = ([1] + [__t2i[tok]
                  for tok in smiles_tokenizer(s)])[:pad_size - 1]
        return 1
    except:
        return 0
    
def encode_test(df):
    """
    Encoder list of smiles to tensor of tokens
    """
    df['pass'] = df['SMILES'].apply(get_token_zdx)
    res_df = df[df['pass']==1]
    pro_df = df[df['pass']==0]
    l_fail = len(df[df['pass']==0])
    del res_df['pass']
    return res_df, pro_df[['ID', 'SMILES', 'plogP']], l_fail

def decode(tokens_tensor):
    """
    Decodes from tensor of tokens to list of smiles
    """

    smiles_res = []

    for i in range(tokens_tensor.shape[0]):
        cur_sm = ''
        for t in tokens_tensor[i].detach().cpu().numpy():
            if t == 2:
                break
            elif t > 2:
                cur_sm += __i2t[t]
        smiles_res.append(cur_sm)
    return smiles_res

def get_vocab_size():
    return len(__i2t)
