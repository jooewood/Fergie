- [1. Create the base docker image](#1-create-the-base-docker-image)
- [2. How to run scripts](#2-how-to-run-scripts)
  - [2.1. Preprocess data](#21-preprocess-data)
  - [2.2. Train model](#22-train-model)
  - [2.3. Generate molecule from trained model](#23-generate-molecule-from-trained-model)
  - [2.4. Preprocess data + Train model + Generate molecule from trained model](#24-preprocess-data--train-model--generate-molecule-from-trained-model)

# 1. Create the base docker image

- Running "make" in this folder creates the base docker image, docker.yfish.x/fergie, with all dependent packages.

```bash
11:16 zdx@M1:~/src/fergie$ make
```

# 2. How to run scripts

## 2.1. Preprocess data

- Preprocess data: HPK1.csv, zinc.csv, kinase.csv, non_kinase.csv  
- **Parameters:**
  ``-i``: A compound library folder path contains the input files which will be preprocessed.  
  ``-f``: Files which will be preprocessed.  
  ``-o``: Folder path to save the preprocessed molecules.  

```bash
09:00 zdx@M1:~$ docker run --gpus all --mount type=bind,src=/y/Aurora/Fergie/data/raw/,dst=/data --mount type=bind,src=/y/home/zdx/project/fergie/,dst=/output docker.yfish.x/fergie main.py preprocess -i /data/ -f HPK1.csv zinc.csv kinase.csv non_kinase.csv -o /output/data/
```

## 2.2. Train model

- **Parameter:**  
  ``-i``: Folder path contains the input files.  
  ``-o``: Folder path saves the trained model and generated molecules.  
  ``-a``: File contains inhibitors of the specific target.  
  ``-f``: File contains inhibitors of the same family of the specific target.  
  ``-n``: File contains inhibitors of different family of the specific target.  
  ``-m``: FIle contains small molecules have smiliar properties to the inhibitors of the specific target.  
  ``-b``: batch size, default is 500.

```bash
09:18 zdx@M1:~$ docker run --gpus all --mount type=bind,src=/y/home/zdx/project/fergie/,dst=/fergie docker.yfish.x/fergie main.py train -i /fergie/data/ -o /fergie/output/HPK1 -a HPK1_preprocess.csv -f kinase_preprocess.csv -n non_kinase_preprocess.csv -m zinc_preprocess.csv -b 1642
```

## 2.3. Generate molecule from trained model

- After training model, we can sample molecules from it.
- **Parameters:**  
  ``-i``: Location of model saved.  
  ``-o``: Location to save the generated molecule.  
  ``-p``: The name of the target.  
  ``-l``: Dimension of latent code size.  
  ``-n``: Total number of molecule to sample.  
  ``-b``: Number of molecule to sample in a batch.  
  ``-g``: GPU use or not.  
  ``--gpu``: Which GPU to use.  

```bash
02:33 zdx@M1:~$ docker run --gpus all --mount type=bind,src=/y/home/zdx/project/fergie/,dst=/fergie docker.yfish.x/fergie main.py initial_sampling -i /fergie/output/HPK1/model/0.5_100/ -o /fergie/output/HPK1/ -p HPK1
```

## 2.4. Preprocess data + Train model + Generate molecule from trained model

- Combine the processes of preprocessing data, training model and generating molecule from trained model.  
- **Parameters:**
  ``-i``: A compound library folder path contains the input files which will be preprocessed.  
  ``-p_f``: Files which will be preprocessed.  
  ``-p_o``: Folder path to save the preprocessed molecules.  
  ``-o``: Folder path saves the trained model and generated molecules.  
  ``-a``: File contains inhibitors of the specific target.  
  ``-f``: File contains inhibitors of the same family of the specific target.  
  ``-n``: File contains inhibitors of different family of the specific target.  
  ``-m``: FIle contains small molecules have smiliar properties to the inhibitors of the specific target.  
  ``-t``: Set the token length for the model to learn.  
  ``-d``: Control the size of distribution.  
  ``-g``: GPU use or not.  
  ``--gpu``: Which GPU to use.  
  ``-c``: Set the number of cpu to load data during the training.  
  ``--protein``: The name of the target.  
  ``--sampling_latent_size``: Dimension of latent code size.  
  ``--num_sample``: Total number of molecule to sample.  
  ``--sampling_batch_size``: Number of molecule to sample in a batch.

```bash
02:33 zdx@M1:~$ docker run --gpus all --mount type=bind,src=/y/Aurora/Fergie/data/raw/,dst=/data --mount type=bind,src=/y/home/zdx/project/fergie/,dst=/output docker.yfish.x/fergie main.py preprocess_train_initialsampling -i /data/ -p_f HPK1.csv zinc.csv kinase.csv non_kinase.csv -p_o /output/data/ -o /output/output/HPK1 -a HPK1_preprocess.csv -f kinase_preprocess.csv -n non_kinase_preprocess.csv -m zinc_preprocess.csv --train_batch_size 1642
```
