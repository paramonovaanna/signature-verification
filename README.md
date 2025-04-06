# Offline signature verification using deep learning

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This project focuses on offline signature verification using deep learning models ConvNeXt and CoAtNet, that have been pretrained on ImageNet-21k. It implements standalone and siamese training pipelines and several methods of signature verification, including CatBoost Classificator. We use GPDS_Synthetic, CEDAR and UTSig datasets.

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. To use GPDS-Synthetic you have to follow the instructions in this link: https://gpds.ulpgc.es/downloadnew/download.htm

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model using output layer or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
To use different methods of classification:

```bash
python3 classify.py HYDRA_CONFIG_ARGUMENTS
```

To train catboost classificator:

```bash
python3 train_catboost.py HYDRA_CONFIG_ARGUMENTS
```

You can choose different training pipelines by changing "mode" argument in hydra to "siamese" or "standalone" and configure different dataset partitions by choosing "pairs", "singles" or "triplets" data.modes in hydra.

## Examples

To train ConvNeXt-T on CEDAR, run:

```bash
python3 train.py model=baseline
```

If you want train CoAtNet-L using siamese networks on UTSig, run this instead:

```bash
python3 train.py model=baseline model=coatnet-large datasets=utsig mode="siamese"
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
