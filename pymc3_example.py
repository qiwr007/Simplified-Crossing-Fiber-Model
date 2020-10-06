import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

warnings.simplefilter(action="ignore", category=FutureWarning)

#%config InlineBackend.figure_format = 'retina'
az.style.use('arviz-darkgrid')
print(f'Running on PyMC3 v{pm.__version__}')
print(f'Running on ArviZ v{az.__version__}')


with pm.Model() as model:
    x = pm.Normal("x", mu=0, sigma=1, shape=100)
    idata = pm.sample(cores=4, return_inferencedata=True)

#az.plot_energy(idata);