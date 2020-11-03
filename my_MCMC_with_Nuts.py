import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np

if __name__ == '__main__':
    my_model = pm.Model()
    with my_model:
        # build up prior distribution for f1, phi_1, phi_2 and sigma^-2
        f1 = pm.Uniform("f1", lower=0, upper=1)
        phi_prime = pm.Uniform("phi_prime_1", lower=0, upper=np.pi, shape=2)
        # phi_prime_2 = pm.Uniform("phi_prime_2", lower=0, upper=np.pi)
        sigma_exp = pm.InverseGamma("sigma^-2", alpha=200, beta=1)

        trace = pm.sample()
        # step = pm.NUTS()
        # trace = pm.sample(2000, tune=1000, init=None, step=step, cores=2)

    az.plot_trace(trace)
    plt.show()