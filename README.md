# Code for "Distilled importance sampling" paper

## Requirements

The following package versions were used for the main run of the code:

matplotlib==3.1.1
numpy==1.10.4
pandas==0.25.1
scipy==0.16.1
seaborn==0.9.0
tensorflow==1.13.2
tensorflow-probability==0.6.0

PYTHON VERSION?

## How to run

The experiments in the paper can be reproduced (up to random seeds) using the scripts

* `sin_example.py`
* `MG1_comparison.py`
* `Lorenz_example.py`

**Warning** the Lorenz example code still has a hard-to-reproduce bug, and occasionally crashes.

Other scripts included are

* `MG1_example.py` - runs a single analysis for the MG1 example - IS THIS UP-TO-DATE? E.G. M CHOICE
* `MG1_example_sumstats.py` - runs a single analysis for the MG1 example using quantile summaries
* `MG1_post_plots.py` - creates plots for the MG1 example. This uses MCMC output in the MATLAB file `paper_1_1_1_16_1.mat`.
* `Lorenz_data.py` - generates a dataset for the Lorenz model

## Implementation details

To run DIS you should create a DIS object, as defined in `DIS.py`,
and supply this with a model object with the following methods:

LIST

Models for the applications in the paper are defined in LIST.

Also note that `SDE.py` contains classes for discretised SDEs for the Lorenz example,
including SDEs with learnable drift and diffusion.
Principally this is code to sample from them and calculate densities.

## NOTES TO SELF

`sin_example.py` WORKS AS-OF 14TH NOV

`Lorenz_data.py` WORKS AS-OF 1TTH NOV

`MG1_comparison.py`, `Lorenz_example.py` START RUNNING OK, DIDN'T CHECK FULL RUN YET

