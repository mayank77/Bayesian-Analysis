#Dataset/Context Refer : Bioassay - "Bayesian Data Analysis, Third Edition - Aki Vehtari"

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = (-.863, -.296, -.053, .7270)
n = (5, 5, 5, 5)
y = (0, 1, 3, 5)

def logitinv( x ):
    return 1/(1+np.exp(-x))

def prior( a,b ):
    return np.exp(-1/1.5*((a*a)/4+((b-10)*(b-10))/100-a*(b-10)/20))

def likelihood( a,b ):
    tmp=1
    for i in range(1,len(x)):
        tmp = tmp*logitinv(a+b*x[i])**y[i]*(1.0-logitinv(a+b*x[i]))**(n[i]-y[i])
    return tmp

def pd( a,b ):
    return prior(a,b)*likelihood(a,b);

#PART A1 - POSTERIOR DENSITY AT A GRID OF POINTS

alpha = np.linspace(-5.0000,10.0000,2500)
beta = np.linspace(-5.0000,30.0000,2500)

na = len(alpha)
nb = len(beta)
z = np.ndarray((na, nb))
z_test = np.ndarray((1000, 1000))
for i in range(1,na):
    z[i] = pd(alpha, beta[i])

#PART A2 - 1000 SAMPLE POINTS    
alpha_sample = np.ndarray(1000)
beta_sample= np.ndarray(1000)
for i in range (0,1000):
    sample = rnd.randint(0,2499)
    alpha_sample[i]=(alpha[sample])
    sample1 = rnd.randint(0,2499)
    beta_sample[i]=(beta[sample1])
    
for i in range(1,1000):
    z_test[i] = pd(alpha_sample, beta_sample[i])
  
#PART A3 - POSTERIOR CONTOUR PLOT FOR THE PARAMETERS

plt.contour(alpha,beta,z,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the posterior distribution")
plt.show()
#PART A4 - POSTERIOR SCATTER PLOT FOR THE PARAMETERS

x, y = np.meshgrid(alpha_sample,beta_sample)
plt.scatter(x,y,z_test)
plt.show()
#PART A5 - LD50 DISTRIBUTION

bpi = beta_sample > 0
samp_ld50 = -alpha_sample[bpi]/beta_sample[bpi]

plt.subplot(3,1,3)
plt.hist(samp_ld50, np.arange(-0.5, 0.5, 0.02))
plt.xlim([-0.5, 0.5])
plt.xlabel(r'LD50 = -$\alpha/\beta$')
plt.yticks(())
plt.show()

#PART B1 - PRIOR DISTRIBUTION

z1 = np.ndarray((na, nb))
for i in range(1,na):
    z1[i] = prior(alpha, beta[i])
plt.contour(alpha,beta,z1,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the prior distribution")
plt.show()

#PART B1 - LIKELIHOOD DISTRIBUTION

z2 = np.ndarray((na, nb))
for i in range(1,na):
    z2[i] = likelihood(alpha, beta[i])
plt.contour(alpha,beta,z2,21)
plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("The contour plot of the likelihood distribution")
plt.show()
