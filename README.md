# Code for "Distilling importance sampling" paper

This code is to reproduce the experiments in the paper [Distilling importance sampling](https://arxiv.org/abs/1910.03632).

## Versions

We ran the code using python 3.6.8 and the following package versions:

matplotlib==3.1.1  
numpy==1.10.4  
pandas==0.25.1  
scipy==0.16.1  
seaborn==0.9.0  
tensorflow==1.13.2  
tensorflow-probability==0.6.0

## Contents

### Experiment scripts

The distilled importance sampling experiments in the paper can be reproduced (up to random seeds) using the scripts

* `Lorenz_example1.py`
* `Lorenz_example2.py`
* `MG1_comparison.py`
* `sin_example.py`

**Warning** the Lorenz example code occasionally crashes due to a rare hard-to-reproduce bug.

### Miscellaneous

* `DIS.py` - code for distilled importance sampling algorithm
* `Lorenz_data.npy` - dataset used in Lorenz examples
* `Lorenz_data.py` - code used to generate the Lorenz dataset
* `Lorenz_functions.py` - shared functions for the Lorenz examples
* `SDE.py` - classes for discretised SDEs for the Lorenz example

### Plots

These scripts create plots used in the paper. Note that they require various results files.

* `MG1_post_plots.py` - plots for the MG1 example results
* `pmcmc/Lorenz_ex1_plots.py` - plots for Lorenz example 1 (i.e. case with unknown sigma)

### PMCMC

This R code uses the `pomp` package.

* `pmcmc/loglike_tuning.R` - To tune the number of particles in both Lorenz examples
* `pmcmc/Lorenz63_ex1_pomp.R` - Runs PMCMC for Lorenz example 1

### Results

Various output files from the runs of the experiments presented in the paper.

* `results/Lorenz63_ex1_mcmc.csv` - PMCMC output for Lorenz example 1
* `results/Lorenz63_ex1_pomp.Rout` - shell output for PMCMC analysis of Lorenz example 1
* `results/Lorenz63_ex1.out` - shell output for DIS analysis of Lorenz example 1
* `results/Lorenz63_ex2.out` - shell output for DIS analysis of Lorenz example 2
* `results/Lorenz63_example1_pars.npy` - parameter sample from DIS analysis of Lorenz example 1 (resampled IS output)
* `results/Lorenz63_example2_pars.npy` - parameter sample from DIS analysis of Lorenz example 2 (resampled IS output)
* `results/paper_1_1_1_16_1.mat` - MCMC output for MG1 example

## Implementation details

To run DIS you should create a DIS object, as defined in `DIS.py`.
The inputs are described in the python docstring documentation.
Here we elaborate on some aspects of one input: `model`.
The model object should have the following methods:

`likelihood_prelims(self, inputs)`  
`log_tempered_target(self, eps)`

An iteration of DIS samples some parameter values and evaluates their importance weights.
This is generally done several times for different values of `eps`.
Rather than redoing all the calculations each time, shared calculation for these evaluations are just performed once, by `likelihood_prelims`, with `inputs` containing the sampled parameter values.
Examples of shared calculations are evaluating the likelihood or performing simulator runs.
The results are then reused when `log_tempered_target` is called.

The model object must also have a `max_eps` variables, giving the initial epsilon value.
