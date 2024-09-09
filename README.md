# Fe_alloy_phaseshift

Install all relevant libraries.
I used python 3.10. You should be able to install all relavant packages using:
"""
conda create -n "matminer-Ax" python=3.10
conda install jupyterlab matplotlib
conda install pytorch torchvision -c pytorch
pip install matminer
pip install ax-platform
pip install pymc
pip install corner
"""

Check also requirements.txt


Code deliverables 

1. data_analysis.ipynb: explores given datasets and calculate statistical values.
2. fit_GP.ipynb: trains GPR models for Bs and Tc prediction. Steps includes data cleanup, model definition, and model training. This code generates Figure 2.
3. fit_GP_elemental.ipynb: trained a reduced feature space surrogate model, then predict values along Fe composition space. This code generates Figure 3.
4. full_cycle-3p.ipynb and Ax_service-3p.ipynb: implements MOBO using Ax’s dev and service API accordingly for 3D parameter space with Cu, Nb elemental composition constrained. Particular steps include full feature-space surrogate model training, experiment definition (surrogate, acquisition, and run strategy), MOBO loop then Bayesian inference to obtain the Pareto front. This code generates Figure 4. 
5. Ax_service-5p.ipynb: my implementations of MOBO using Ax’s service API. This code generates Figure 1b,c and Figure 5.
6. Bayesian_inference.ipynb: for uncertainty quantification and sensitivity analysis. Particular steps includes training a reduced feature space surrogate model, infer posterior distribution of Fe composition within a subset of the dataset, then predict values along distribution.  This code generates Figure 6.
7. utils folder contain utility functions used throughout these notebooks. The library written by me offers a powerhouse used throughout the simulations and analyses. 
