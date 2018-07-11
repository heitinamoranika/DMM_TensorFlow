import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import*
from spectralradius import*
from local_optimizer_Adam import*
from homotopy_optimizer_Possion import*
from multigrid_two_level import*
m = 8


def u_real(x): return x*(1-x)*np.exp(x)


def f(x): return (3*x+x**2)*np.exp(x)


s = 2
GMM_OUTPUT = Multigrid_circle(m, Possion(
    m), interpolate(m), restriction(m), s, 2/3,  f, 20)
R, P, w, GMMradius, DMMradius = Local_optimizer_Adam(
    m, Possion(m), interpolate(m), restriction(m), 2/3, 50, 100)
GMMradius = RadiusNumpy(m, Possion(m), interpolate(m), restriction(m), 2/3, s)
DMMradius = RadiusNumpy(m, Possion(m), P, R, w, s)
DMM_OUTPUT = Multigrid_circle(m, Possion(m), P, R, s, w, f, 20)
X = np.linspace(0, 1, len(GMM_OUTPUT))
REAL_OUTPUT = u_real(X)
plt.figure(1)
plt.plot(X, GMM_OUTPUT, label='GMM_OUTPUT')
plt.plot(X, DMM_OUTPUT, label='DMM_OUTPUT')
plt.plot(X, REAL_OUTPUT, label='Real_OUTPUT')
plt.legend(loc='upper left')
plt.show()
print(GMMradius)
print(DMMradius)
