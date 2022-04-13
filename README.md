# Determinantal Point Process Dropout

This project was done as part of my Random Matrix Theory class project. 

In this repo I implement dropout using determinantal point processes (DPP) to select which neurons are dropped. I implement DPP sampling algorithms as well as consider a similar linear regression problem for insight into why DPP dropout may be more effective than the standard poisson sampled dropout.


#### Reproducibility 

The code to reproduce the figures in the paper along with other miscellaneous experiments with DPPs is available in the "Notebook Experiments (DPP Sampling + Regression Study)" notebook.



You may run the experiments seen in my report by running 

```
./run_dropout_comparisons.sh
```
