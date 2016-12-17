#Inference for difference between proportions
#noninformative or weakly informative prior distribution

import numpy as np;
from scipy.stats import beta
import matplotlib.pyplot as plt

cn = 674
cd = 39
cp = cd/cn
tn = 680
td = 22
tp = td/tn
A=1
B=1

mean_c = (cd+A)*(1.0000000)/(A+B+cn);
mode_c = (cd+A-1)*(1.0000000)/(A+B+cn-2);

Y = np.linspace(0,1)
posterior_c = beta.pdf (Y,cd+A,cn+B-cd)
plt.plot(Y, posterior_c )

mean_t = (td+A)*(1.0000000)/(A+B+tn);
mode_t = (td+A-1)*(1.0000000)/(A+B+tn-2);

Y = np.linspace(0,1)
posterior_t = beta.pdf (Y,td+A,tn+B-td)
plt.plot(Y, posterior_t )

p_c = np.random.beta(cd+A,cn+B-cd,1000)
p_t = np.random.beta(td+A,tn+B-td,1000)
p_final = []
for i in range(0,1000):
    p_final.append( (p_t[i]/(1-p_t[i]))/(p_c[i]/(1-p_c[i])) )
plt.hist(p_final,bins=Y)
range_ppf = []
range_ppf.append(np.percentile(p_final,0.025))
range_ppf.append(np.percentile(p_final,0.975))
