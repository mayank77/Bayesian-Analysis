'''
Assess predictive performance of the pooled/separate/hierarchical Gaussian models using PyStan
'''

import pystan
import psis as ps
import numpy as np

factory = np.loadtxt('factory.txt')

pool_code = """
data{
int<lower=0> N; // number of data points
vector[N] y;
}
parameters{
real mu;
real<lower=0> sigma;
}
model{
y ~ normal_log(mu, sigma);
}
generated quantities{
real ypred;
vector[N] log_lik;
ypred = normal_rng(mu, sigma);
for (n in 1:N)
    log_lik[n] <- normal_log(y[n], mu, sigma);
}
"""
pool_data = {'N': len(factory.flatten()),'y': factory.flatten()}
pool_fit = pystan.stan(model_code=pool_code, data=pool_data)
samples = pool_fit.extract(permuted=True)
loo, loos, ks = ps.psisloo(samples['log_lik'])
lppd = 0
for i in range (0,len(loos)):
    lppd = lppd + np.log( np.mean( np.exp(samples['log_lik'][:,i]) ) )
peff = lppd - loo

separate_code = """
data{
int<lower=0> N; // number of data points
int<lower=0> K; // number of groups
int<lower=1, upper=K> x[N]; // group indicator
vector[N] y;
}
parameters{
vector[K] mu; // mean for each group
real<lower=0> sigma; // common variance
}
model{
for (n in 1:N)
y[n] ~ normal_log(mu[x[n]], sigma);
}
generated quantities{
vector[N] log_lik;
for (n in 1:N)
    log_lik[n] <- normal_log(y[n], mu, sigma);
}
"""
separate_data = {'N': len(factory.flatten()),'K': 6, 'x': np.tile(np.arange(1,7),
factory.shape[0]), 'y': factory.flatten(),}
sepa_fit = pystan.stan(model_code=separate_code, data=separate_data)
sepa_mean = sepa_fit.extract(permuted=True)['mu'][:,-1]
samples1 = sepa_fit.extract(permuted=True)
loo, loos, ks1 = ps.psisloo(samples1['log_lik'])
lppd = 0
for i in range (0,len(loos)):
    lppd = lppd + np.log( np.mean( np.exp(samples1['log_lik'][:,i]) ) )
peff1 = lppd - loo

hierarchical_code = """
data{
int<lower=0> N; // number of data points
int<lower=0> K; // number of groups
int<lower=1, upper=K> x[N]; // group indicator
vector[N] y;
}
parameters{
real mu0; // hyper-prior mean
real<lower=0> tau; // hyper-prior std
vector[K] mu; // group means
real<lower=0> sigma; // group common std
}
model{
mu ~ normal(mu0, tau);
for (n in 1:N)
y[n] ~ normal_log(mu[x[n]], sigma);
}
generated quantities{
vector[N] log_lik;
for (n in 1:N)
    log_lik[n] <- normal_log(y[n], mu, sigma);
}
"""
hierarchical_data = {'N': len(factory.flatten()),
'K': 6,
'x': np.tile(np.arange(1,7), factory.shape[0]),
'y': factory.flatten()}
hier_fit = pystan.stan(model_code=hierarchical_code, data=hierarchical_data)
samples2 = hier_fit.extract(permuted=True)
loo, loos, ks2 = ps.psisloo(samples2['log_lik'])
lppd = 0
for i in range (0,len(loos)):
    lppd = lppd + np.log( np.mean( np.exp(samples2['log_lik'][:,i]) ) )
peff2 = lppd - loo
