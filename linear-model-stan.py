'''
Linear Model Using Stan
'''

import pystan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as sp
from scipy.stats import t

plt.rc('font', size=14)
plt.rc('lines', color='#377eb8', linewidth=2)
plt.rc('axes', color_cycle=('#377eb8','#e41a1c','#4daf4a',
                            '#984ea3','#ff7f00','#ffff33'))
                            

hier_code = """
data {
    int<lower=0> N; // number of data points 
    int<lower=0> K; // number of groups 
    int<lower=1,upper=K> x[N]; // group indicator 
    vector[N] y; // 
}
parameters {
    real mu0;             // prior mean 
    real<lower=0> sigma0; // prior std 
    vector[K] mu;         // group means 
    real<lower=0> sigma;  // common std 
}
model {
    mu0 ~ normal(10,10);      // weakly informative prior 
    sigma0 ~ cauchy(0,4);     // weakly informative prior 
    mu ~ normal(mu0, sigma0); // population prior with unknown parameters
    sigma ~ cauchy(0,4);      // weakly informative prior
    for (n in 1:N)
      y[n] ~ normal(mu[x[n]], sigma);
}
"""


# Data for Stan
data_path = 'factory.csv'
d = np.loadtxt(data_path, dtype=np.double, delimiter=',', skiprows=0)
x = np.tile(np.arange(1,7), d.shape[0]) 

y = d[:,0:6].ravel()

N = len(x)
data = dict(
    N = N,
    K = 6,  # 3 groups
    x = x,  # group indicators
    y = y   # observations
)

# Compile and fit the model
fit = pystan.stan(model_code=hier_code, data=data)

# Analyse results
samples = fit.extract(permuted=True)
print "std(mu0): {}".format(np.std(samples['mu0']))
mu = samples['mu']
# Matrix of probabilities that one mu is larger than other
ps = np.zeros((3,3))
for k1 in range(3):
    for k2 in range(k1+1,3):
        ps[k1,k2] = np.mean(mu[:,k1]>mu[:,k2])
        ps[k2,k1] = 1 - ps[k1,k2]
print "Matrix of probabilities that one mu is larger than other:"
print ps
# Plot
plt.boxplot(mu)
plt.show()


nu = len (zip(*d))-1
s = np.std(zip(*d)[5],ddof=1)
m = np.mean(zip(*d)[5])
interval_a_interval = t.interval(0.95,nu,m,s/np.sqrt(nu+1))
interval_b_interval = t.interval(0.95,nu,m,s*np.sqrt(1+(1/(nu+1))))

test = np.linspace(50,150)
p1 = plt.plot(test,t.pdf(test,nu,m,scale =s/np.sqrt(nu+1)), linestyle='--')
p2 = plt.plot(test,t.pdf(test,nu,m,scale=s*np.sqrt(1+(1/(nu+1)))),linestyle= '-')

s = np.std(zip(*d),ddof=1)
m = np.mean(zip(*d))
interval_c_interval = t.interval(0.95,nu,m,s/np.sqrt(nu+1))
p3 = plt.plot(test,t.pdf(test,nu,m,scale =s/np.sqrt(nu+1)), linestyle=':')
plt.legend([p1,p2,p3],["Part (A)" , "Part (B)", "Part (C)" ] )
plt.show()
