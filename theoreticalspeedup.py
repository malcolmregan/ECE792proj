import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

Nvals = np.logspace(1,6,100).tolist()
kvals = np.logspace(0,3,100).tolist()

classO = np.zeros(shape=(len(Nvals),len(kvals)))
quanO = np.zeros(shape=(len(Nvals),len(kvals)))

s = 0.5
eps = .01

figclass = plt.figure()
axclass = figclass.add_subplot(111)
figquan = plt.figure()
axquan = figquan.add_subplot(111)
figspeedup = plt.figure()
axspeedup = figspeedup.add_subplot(111)

for i,N in enumerate(Nvals):
    for j,k in enumerate(kvals):
        classO[i,j] = k*s/(-np.log(eps))*N*np.log2(N) 
        quanO[i,j] = k**2*s**2/eps*np.log2(N)
speedup = classO/quanO 

cmin = np.min(classO)
qmin = np.min(quanO)
cmax = np.max(classO)
qmax = np.max(quanO)

minscale = np.min([cmin, qmin])
maxscale = np.max([cmax, qmax])

extent = [np.min(kvals),np.max(kvals),np.min(Nvals),np.max(Nvals)]

c = axclass.imshow(classO, origin='lower',aspect='auto',norm=LogNorm(minscale, maxscale),extent=extent)
ccbar = figclass.colorbar(c)
axclass.set_xscale('log')
axclass.set_yscale('log')
axclass.set_title('Computational Complexity for Classical LA Solver')
axclass.set_xlabel('$\kappa$')
axclass.set_ylabel('$N$')

q = axquan.imshow(quanO, origin='lower', aspect='auto',norm=LogNorm(minscale, maxscale), extent=extent)
qcbar = figquan.colorbar(q)
axquan.set_xscale('log')
axquan.set_yscale('log')
axquan.set_title('Computational Complexity for HHL \n (after |b> is prepared and unitary A is implemented))')
axquan.set_xlabel('$\kappa$')
axquan.set_ylabel('$N$')


s = axspeedup.imshow(speedup, origin='lower', aspect='auto',norm=LogNorm(), extent=extent)
scbar = figspeedup.colorbar(s)
axspeedup.set_title('Theoretical Speedup $\\frac{O(c)}{O(q)}$')
axspeedup.set_xscale('log')
axspeedup.set_yscale('log')
axspeedup.set_xlabel('$\kappa$')
axspeedup.set_ylabel('$N$')
plt.show()
