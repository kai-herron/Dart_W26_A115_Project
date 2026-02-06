import nuc_burn
import numpy as np

T = np.linspace(1.5e7, 2e7, 100)
Rho = np.linspace(1.5e2, 1.5e2, 100)
Time=1000
e, mu, mf = nuc_burn.burn(T, Rho, Time)

print(e)
print(mu)
print(mf)

# code to compute compositions
Hspec = ['H-1','H-2','H-3']
Hespec = ['He-3','He-4']

X = []
for H in Hspec:
    X.append(mf.getMassFraction(H))

Y = []
for He in Hespec:
    Y.append(mf.getMassFraction(He))

X = np.sum(X)
Y = np.sum(Y)

Z = 1 - (X+Y)
print('X:', X)
print('Y:', Y)
print('Z:', Z)
print('done')