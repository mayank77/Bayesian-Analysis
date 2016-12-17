#Dataset BIOASSAY reference : "Bayesian Data Analysis, Third Edition - Aki Vehtari"

import numpy as np
from scipy.stats import distributions
import matplotlib.pyplot as plt
import scipy.stats as sp

dbin = distributions.binom.logpmf
dnorm = distributions.norm.logpdf
rnorm = np.random.normal
runif = np.random.rand

log_dose = [-.86,-.3,-.05,.73]
n = 5
deaths = [0, 1, 3, 5]

ld50 = lambda alpha, beta: -alpha/beta
invlogit = lambda x: 1/(1. + np.exp(-x))

dbinom = distributions.binom.logpmf
dnorm = distributions.norm.logpdf

def bioassay_posterior(alpha, beta):
    logp = dnorm(alpha, 0, 10000) + dnorm(beta, 0, 10000)
    p = invlogit(alpha + beta*np.array(log_dose))
    logp += dbinom(deaths, n, p).sum()
    return logp

def metropolis_bioassay(n_iterations, initial_values, prop_var=1,
                     tune_for=None, tune_interval=100):
    n_params = len(initial_values)
    prop_sd = [prop_var] * n_params
    trace = np.empty((n_iterations+1, n_params))
    trace[0] = initial_values
    accepted = [0]*n_params
    current_log_prob = bioassay_posterior(*trace[0])
    if tune_for is None:
        tune_for = n_iterations/2
    for i in range(n_iterations):
        if not i%1000: print('Iteration', i)
        current_params = trace[i]
        for j in range(n_params):
            p = trace[i].copy()
            theta = rnorm(current_params[j], prop_sd[j])
            p[j] = theta
            proposed_log_prob = bioassay_posterior(*p)
            alpha = proposed_log_prob - current_log_prob
            u = runif()
            if np.log(u) < alpha:
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                accepted[j] += 1
            else:
                trace[i+1,j] = trace[i,j]
            if (not (i+1) % tune_interval) and (i < tune_for):
                acceptance_rate = (1.*accepted[j])/tune_interval
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1
                accepted[j] = 0

    return trace[tune_for:], accepted

tr, acc = metropolis_bioassay(10000, (0,0))

for param, samples in zip(['intercept', 'slope'], tr.T):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(samples)
    axes[0].set_ylabel(param)
    axes[1].hist(samples[len(samples)//2:])

a, b = tr.T
print('LD50 mean is {}'.format(ld50(a,b).mean()))

sample_a = tr[:,0]
sample_b = tr[:,1]

plt.scatter(sample_a, sample_b, 10, linewidth=0)

bpi = sample_b > 0
samp_ld50 = -sample_a[bpi] / sample_b[bpi]

plt.hist(samp_ld50, np.arange(-0.5, 0.51, 0.02))
