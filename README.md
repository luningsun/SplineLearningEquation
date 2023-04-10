# splineLearning

This repository contains PyTorch implementation of the submission: Bayesian Spline Learning for Equation Discovery of Nonlinear Dynamics with Quantified Uncertainty.

## 0. Environment Setup

enviroment setup: "conda create --name python36 --file requirements.txt"
conda activate python36
## Cases
It contains two demos cases: Demo/ODE.py and Demo/PDE.py, which coorespond to Lotka Volterra and Burgers'



## 1. Training
To train the ODE demo, run the following command:
```
cd/Demo
python ODE.py
```

To train the PDE demo model on Lung dataset with GraphRNN/DAGG, run the following:

``` 
cd/Demo
python PDE.py                     
```

