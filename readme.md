# Deep Learning-enabled MCMC for Probabilistic State Estimation in District Heating Grids 

This repo contains the code to reproduce the results from the paper "Deep Learning-enabled MCMC for Probabilistic State
Estimation in District Heating Grids", which is published in Applied Energy in 2023.
[https://doi.org/10.1016/j.apenergy.2023.120837](https://authors.elsevier.com/a/1gbdG15eif8FQV) 

## Abstract:
Flexible district heating grids form an important part of future, low-carbon energy systems. We examine probabilistic 
state estimation in such grids, i.e., we aim to estimate the posterior probability distribution over all grid state 
variables such as pressures, temperatures, and mass flows conditional on measurements of a subset of these states. 
Since the posterior state distribution does not belong to a standard class of probability distributions, we use Markov 
Chain Monte Carlo (MCMC) sampling in the space of network heat exchanges and evaluate the samples in the grid state 
space to estimate the posterior. Converting the heat exchange samples into grid states by solving the non-linear grid 
equations makes this approach computationally burdensome. However, we propose to speed it up by employing a deep neural 
network that is trained to approximate the solution of the exact but slow non-linear solver. This novel approach is 
shown to deliver highly accurate posterior distributions both for classic treeshaped as well as meshed heating grids, 
at significantly reduced computational costs that are acceptable for online control. Our state estimation approach thus
enables tightening the safety margins for temperature and pressure control and thereby a more efficient grid operation.

## About this repo: 
### main files: 
- [MC_sample_generation.ipynb](MC_sample_generation.ipynb): includes code to generate MC-samples consisting of random 
heat powers and corresponding grid states. -> this has to run once to generate training data for the DNN
- [train_DNN.ipynb](train_DNN.ipynb): includes code to train the neural network approximating the classical NR solver 
    -> this has to run only once 
- [MCMC_interference.ipynb](MCMC_interference.ipynb): includes the code for the MCMC-interference.
    ->  this has to run again, each time a new measurement is gathered. 
- [approximate_steady_state_error.ipynb](approximate_steady_state_error.ipynb): approximation of the steady state error as discussed in Appendix C

 ### folders: 
 - [data_files](data_files): 
    - [Library_Pipe_Parameters.xlsx](data_files/Library_Pipe_Parameters.xlsx): pipe parameter used, e.g. diameter and isolation
    - [inputs_loop](data_files/inputs_loop): contains the grid topology and the demand specifications for the loop-testcase
    - [results_loop](data_files/results_loop): contains random demand inputs and correspoding grid states as .csv files.    
    The samples are used to train the DNN and as the first step of the MC-SIR algorithm. 
    - *disclaimer: this repo does not include the data files for the gird_tree testcase, since these are confidential*
 - [libdnnmcmc](libdnnmcmc): This folder includes library files and helper functions
    - [evaluation_functions.py](libdnnmcmc/evaluation_functions.py): functions to evaluate results
    - [importance_sampling.py](libdnnmcmc/importance_sampling.py): Code for the SIR-step of the MC-SIR baseline
    - [se_NN_lib.py](libdnnmcmc/se_NN_lib.py): custom layers, losses and metrics for the neural network 
    - [state_equations.py](libdnnmcmc/state_equations.py): Object summarising the state equations and defining useful functions for the solver to interact 
      with these equations. The object defined in this file is a central piece used almost everywhere in the code,  
      usually referenced as "SE"
    - [steady_state_solvers.py](libdnnmcmc/steady_state_solvers.py): contains a "fixpoint_iterator" to presolve the state equations and a NR-algorithm for the 
      faster convergence close to the solution. Defines a SE_solver() object to choose the right solver automatically. 
    - [utility.py](libdnnmcmc/utility.py): helper functions for data import and general definitions such as the ZeroTruncatedNormal distribution. 
 - [models](models): The trained neural networks will be saved here
 - [results](results): the evaluation of the MCMC approximation will be saved here
  


