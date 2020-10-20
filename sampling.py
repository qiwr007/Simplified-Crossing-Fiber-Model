import arviz as az
import matplotlib.pyplot as plt
import numpy as np


with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(500)