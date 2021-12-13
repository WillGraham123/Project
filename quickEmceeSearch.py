from keras.models import load_model
from multiprocessing import Pool
import emcee
import os
import time
import numpy as np

nn = load_model("/home/willgraham/Project Work/")

import math

def log_prior(theta):
    Av, As, Ac, Aa, Ap = theta
    if 0 < Av < 50 and -50 < As < 0 and -50 < Ac < 0 and -50 < Aa < 0 and -50 < Ap < 0:
        return 0.0
    return -np.inf
    
def log_prob(p0):
    lp = log_prior(p0)
    if not np.isfinite(lp):
        return -np.inf
    return lp - math.exp(nn.predict(p0.reshape(1, 5)))

nwalkers = 10
ndim = 5
p0 = np.ndarray((nwalkers, ndim), dtype=np.double)

AvField = np.linspace(0, 50, 100)
AsField = np.linspace(-50, 0, 100)
AcField = np.linspace(-50, 0, 100)
AaField = np.linspace(-50, 0, 100)
ApField = np.linspace(-50, 0, 100)

for i in range(nwalkers):
    Av = round(AvField[np.random.randint(0, 100)], 1)
    As = round(AsField[np.random.randint(0, 100)], 1)
    Ac = round(AcField[np.random.randint(0, 100)], 1)
    Aa = round(AaField[np.random.randint(0, 100)], 1)
    Ap = round(ApField[np.random.randint(0, 100)], 1)
    
    p0[i] = [Av, As, Ac, Aa, Ap]

with Pool() as pool:

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(p0, 100, progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))