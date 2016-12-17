#The observed hardness values y1 can be found in file windshieldy1.txt
#Inference for normal mean and deviation
#t-distribution

import numpy as np;
from scipy.stats import t
import matplotlib.pyplot as plt
data=np.genfromtxt("windshieldy1.txt")
nu = len(data)-1
s = np.std(data,ddof=1)
v = np.var(data,ddof=1)
m = np.mean(data)
interval_a_part = t.interval(0.95, nu, m, s / np.sqrt(nu+1))
interval_b_part = t.interval(0.95, nu, m, s*np.sqrt(1+(1/(nu+1))))
x = np.linspace(12, 18)
p1,=plt.plot(x, t.pdf(x, nu, m, scale=s / np.sqrt(nu+1)), linestyle='--')
plt.xlabel('X')
plt.ylabel('Distribution Value')
p2,=plt.plot(x, t.pdf(x, nu, m, scale=s*np.sqrt(1+(1/(nu+1)))), linestyle='-')
plt.legend([p1, p2], ["Part(A)", "Part(B)"])
plt.show()
aa=t.mean(nu, m, s*np.sqrt(1+(1/(nu+1))))
