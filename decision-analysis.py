'''
Decision Analysis Based On Utility using PyStan
'''

import numpy as np
import matplotlib.pyplot as plt
import pystan

factory = np.loadtxt('factory.txt') # factory[y][k]

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
y[n] ~ normal(mu[x[n]], sigma);
}
generated quantities{
real ypred1;
real ypred2;
real ypred3;
real ypred4;
real ypred5;
real ypred6;
real mu7;
ypred1 = normal_rng(mu[1], sigma);
ypred2 = normal_rng(mu[2], sigma);
ypred3 = normal_rng(mu[3], sigma);
ypred4 = normal_rng(mu[4], sigma);
ypred5 = normal_rng(mu[5], sigma);
mu7 = normal_rng(mu0, tau); // ppd of mean for 7th machine
ypred6 = normal_rng(mu[6], sigma); // predictive posterior distribution
}
"""

hierarchical_data = {'N': len(factory.flatten()),
'K': 6,
'x': np.tile(np.arange(1,7), factory.shape[0]),
'y': factory.flatten()}


hier_fit = pystan.stan(model_code=hierarchical_code, data=hierarchical_data)
hier_mean = hier_fit.extract(permuted=True)['mu'][:,-1]
hier_m7 = hier_fit.extract(permuted=True)['mu7']

for j in range(1,7,1):
    EU = 0
    P="ypred"+str(j)
    for i in (hier_fit.extract(permuted=True)[P]):
        if i < 85:
            EU = EU - (100.00/len(hier_fit.extract(permuted=True)[P]))
        if i >= 85:
            EU = EU + (100.00/len(hier_fit.extract(permuted=True)[P]))
    print P, EU

EU = 0
for i in (hier_m7):
    if i < 85:
        EU = EU - (100.00/len(hier_m7))
    if i >= 85:
        EU = EU + (100.00/len(hier_m7))
print EU
