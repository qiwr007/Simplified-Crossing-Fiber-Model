import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


# import theano.tensor as tt

#%config InlineBackend.figure_format = 'retina'

if __name__ == '__main__':
    # Initialize random number generator
    RANDOM_SEED = 8927
    np.random.seed(RANDOM_SEED)
    az.style.use('arviz-darkgrid')

    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
    axes[0].scatter(X1, Y, alpha = 0.6)
    axes[1].scatter(X2, Y, alpha = 0.6)
    axes[0].set_ylabel("Y")
    axes[0].set_xlabel("X1")
    axes[1].set_xlabel("X2")

    basic_model = pm.Model()
    with basic_model:
        # Prior distribution for unknown model parameter
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma = 1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    map_estimate = pm.find_MAP(model=basic_model)
    print(map_estimate)

    with basic_model:
        # instantiate sampler
        step = pm.Slice()

        # draw 5000 posterior samples
        trace = pm.sample(5000, step=step)

    az.plot_trace(trace)
    plt.show()

