# Code for "Distilling importance sampling" paper

This code is to reproduce the experiments in the paper [Distilling importance sampling for likelihood-free inference](https://arxiv.org/abs/1910.03632).

## Version history

* v1.0 is Tensorflow code used in versions 1-3 of the arxiv paper
* v1.1 adds some bug fixes
* v2.0 is PyTorch code for version 4 of the arxiv paper
* V2.1 has some minor modifications to plotting code 
* V2.2 is PyTorch code for version 5 of the arxiv paper
* V2.3 is PyTorch code for version 6 of the arxiv paper

## Python package versions

For the paper the code was run using python 3.8.10 and the following package versions:

* matplotlib 3.5.1
* networkx 2.7.1
* nflows 0.14
* numpy 1.22.1
* pandas 1.4.0
* torch 1.10.2
* UMNN 1.68

## Contents

### General DIS files

* `DIS.py` contains the inference algorithms. The `train` method performs the Distilling Importance Sampling algorithm.
* The `models` directory contains code for various example models. These encapsulate prior + model + tempering scheme. There is also some general code for storing and manipulating weighted samples.
* `utils.py` has some utility functions used in various places in the code.

### DIS experiment scripts

The distilled importance sampling experiments in the paper can be reproduced using the scripts

* `MG1_comparison.py`
* `sin_example.py`
* `SInetwork_example1.py` (this also runs a likelihood-based analysis)
* `SInetwork_example2.py`

Note `.out` files contains terminal output from running the corresponding script. This often includes timing information.

### Other experiments

The `abc` folder contains python code for the ABC-PMC analyses. The experiments can be run using the script `MG1_ABC.py`.

### Plots

These scripts create plots used in the paper. Some other plots are created in within the experiment scripts.

* `MG1_post_plots.py` - plots for the MG1 example results
* `abc/MG1_ABC_plots.py` - plots for ABC baseline on MG1 example

The scripts above require various results files, which are the ad hoc collection of `.npy`, `.pkl`, `.csv` and `.mat` files in the repository.

### Miscellaneous

* `LikelihoodBased.py` and `SInetworkLikelihood.py` help perform likelihood based calculations for the SI network example.