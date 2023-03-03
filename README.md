- [1. A deep generative model for drug design](#1-a-deep-generative-model-for-drug-design)
  - [1.1. Data](#11-data)
    - [1.1.1. Take HPK1 as an example](#111-take-hpk1-as-an-example)
    - [1.1.2. Processing data](#112-processing-data)
  - [1.2. Anaconda](#12-anaconda)
    - [1.2.1. Anaconda common commands](#121-anaconda-common-commands)
  - [1.3. Dependencies](#13-dependencies)
  - [1.4. Usage](#14-usage)
    - [1.4.1. Preprocess](#141-preprocess)
    - [1.4.2. Train fergie](#142-train-fergie)
    - [1.4.3. Generate molecule from trained fergie](#143-generate-molecule-from-trained-fergie)
    - [1.4.4. Filter the initial sampled molecule](#144-filter-the-initial-sampled-molecule)
    - [1.4.5. Chemical space distribution map](#145-chemical-space-distribution-map)
    - [1.4.6. main.py](#146-mainpy)

# 1. A deep generative model for drug design

## 1.1. Data

### 1.1.1. Take HPK1 as an example

These file is located in /y/Fergie/data

|File|Description|
|:---:|:----------|
|zinc.csv|Compound extracted from ZINC database.|
|non_kinase.csv|Common non-kinase inhibitors extracted from ChEMBL database.|
|kinase.csv|Common kinase inhibitors extracted from ChEMBL database.|
|HPK1.csv|Inhibitors extracted from patents.|


### 1.1.2. Processing data

The file is preprocess.py

1. **Remove invalid SMILES string.**
2. **Remove duplicates according to InChi.**
3. **Remove SMILES with invalid token.**
4. **Remove unformatted SMILES string.**
    *  Remove molecules without carbon atoms,it cannot be drug-like molecule. 
    *  Remain the largest one if the molecule contains more than one fragments, which are seperated by '.'.

## 1.2. Anaconda

**RDKit** (wihch can not be installed by pip, so you need install **Anaconda** first, and create a anaconda environment, so all packages below should be installed into the created environment)

```bash
# Download Anaconda
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# Install Anaconda
$ sha256sum Anaconda3-2020.11-Linux-x86_64.sh # Check file integrity
$ bash Anaconda3-2020.11-Linux-x86_64.sh
$ source ~/.bashrc
# After installing anaconda, set the base environment that does not activate CONDA by default
$ conda config --set auto_activate_base false
```

### 1.2.1. Anaconda common commands

|Command|Description|
|:------|:----------|
|conda search "^python$"|Check python version you can install.|
|conda create -n new_env python=3.9.1|Create a new environment with specific python version|
|conda activate new_env|Activate a existed environment|
|conda deactivate|Deactivate current environment|
|conda info --envs|Check the environment you created.|
|conda install -n new_env numpy|Install package into a specific environment|
|conda remove -n new_env --all|Delete a environment|
|python --version| Check the python version in current environment.|
|conda update python| Update the python version in current environment.|
|conda update conda|Update the conda package|
|conda update anaconda|Update the anaconda version|

## 1.3. Dependencies

**RDKit**
Creating a new conda environment with the RDKit installed.

```bash
(base) zdx@M1:~$ conda create -c rdkit -n rdkit-env rdkit
```

Activate the rdkit environment.

```bash
(base) zdx@M1:~$ conda activate rdkit-env
```

Activate:

```bash
(rdkit-env) zdx@M1:~$
```

**PyTorch 1.2.0** [previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)

```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

PyTorch CPU version （if needed）

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**seaborn**

```bash
conda install -c anaconda seaborn
```

**matplotlib**

```bash
conda install -c anaconda matplotlib
```

**MOSES**

```bash
pip3 install molsets
```

**sklearn**

```bash
conda install -c anaconda scikit-learn 
```

**torchsnooper**

```bash
pip3 install torchsnooper
```

**pandas** maybe installed by default.  
**numpy** maybe installed by default.

**Check whether GPU is ready?**

```bash
(rdkit-env) zdx@M1:~$ python
Python 3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

## 1.4. Usage

enter src

### 1.4.1. Preprocess

**parameter:**  
```-i``` : Folder path contains the input files.  
```-f```: Files which will be preprocessed.  
```-o```: Folder path saves the trained model and generated molecules.  

Each file for triaining fergie need through this.  
Data preprocessing of HPK1, ZINC, kinase, non_kinase:  

```bash
./preprocess.py -i /y/Aurora/Fergie/data/raw/ -f HPK1.csv zinc.csv kinase.csv non_kinase.csv -o /y/Aurora/Fergie/data/preprocessed/
```

### 1.4.2. Train fergie

Train fergie need four files.
In TMUX, if you can't use Anaconda Python:  
Deactivate the conda environment, then start tmux, then reactivate the environment inside tmux.

**parameter:**  
```-i``` : Folder path contains the input files.  
```-o```: Folder path saves the trained model and generated molecules.  
```-a```: File contains inhibitors of the specific target.  
```-f```: File contains inhibitors of the same family of the specific target.  
```-n```: File contains inhibitors of different family of the specific target.  
```-m```: FIle contains small molecules have smiliar properties to the inhibitors of the specific target.  
```-b```: batch size, default is 500.

```bash
./train.py -i /y/Aurora/Fergie/data/preprocessed -o /y/Aurora/Fergie/output/HPK1 -a HPK1_preprocess.csv -f kinase_preprocess.csv -n non_kinase_preprocess.csv -m zinc_preprocess.csv -b 1642
```

### 1.4.3. Generate molecule from trained fergie

After training fergie, we can sample molecules from it.

**parameter:**  
```-i```: Location of model saved.  
```-o```: Location to save the generated molecule.  
```-p```: The name of the target.  

```bash
./initial_sampling.py -i /y/Aurora/Fergie/output/HPK1/model/0.5_100/ -o /y/Aurora/Fergie/output/HPK1/ -p HPK1
```

### 1.4.4. Filter the initial sampled molecule

**parameter:**  
```-i``` : File contains compounds will be filtered.  
```-o```: Location to save the filtered molecules.  
```-p```: File contains active compounds for train vae model.  
```-c```: The file contains the property rules to filter compounds.  

```bash
./filter.py -i /y/Aurora/Fergie/output/HPK1/molecule_generated/init_sample_30000.csv -o /y/Aurora/Fergie/output/HPK1/molecule_generated/ -p /y/Aurora/Fergie/preprocessed/HPK1_preprocess.csv -c ../condition/kinase.txt
```

### 1.4.5. Chemical space distribution map

**parameter:**  
```-i```: Input files(please write according to the file size)  
```-o```: output file  
```-l```: figure length  
```-w```: figure width  

```bash
python fergie_umap.py -i /y/Aurora/Fergie/data/preprocessed/kinase_preprocess.csv /y/Aurora/Fergie/data/preprocessed/HPK1_preprocess.csv -o /y/Aurora/Fergie/umap.eps  -l 100 -w 80
```

### 1.4.6. main.py

- This script combines prerocess, train and initial_sampling. The first parameter means the step, and you can choose **preprocess**, **train**, **initial_sampling**, **preprocess_train_initialsampling**
- Different steps have different parameters, for more details of certain step and supported parameters run ``./main.py <step> --help``.

```bash
./main.py preprocess_train_initialsampling -i /y/Aurora/Fergie/gitlab_ci_test/data/ -p_f HPK1.csv zinc.csv -p_o /y/home/zyw/src/fergie/data/ -o /y/home/zyw/project/fergie/output/HPK1 -a HPK1_preprocess.csv -f kinase_preprocess.csv -n non_kinase_preprocess.csv -m zinc_preprocess.csv -d 1000 --train_batch_size 200 -e 10 --protein HPK1 --num_sample 100 --sampling_batch_size 50
```
