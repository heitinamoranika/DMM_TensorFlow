import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import *
from spectralradius import *
from norm import *
from local_optimizer_Adam import *
from homotopy_optimizer_Helmholtz import *
from multigrid_two_level import *

m = 8
s2 = 20
kmax = 50


def f(x): return np.exp(np.pi*x)

NUM_EPOCH = 20
m, GMMradius, DMMradius, R0, P0, w0 = radiusPlot(m, 2, kmax, 5)
GMM_OUTPUT = Multigrid_circle(m, Helmholtz(m, kmax), interpolate(m), restriction(m),2, 2/3,  f, NUM_EPOCH)
DMM_OUTPUT = Multigrid_circle(m, Helmholtz(m, kmax), P0, R0, 2, w0,  f, NUM_EPOCH)

X = np.linspace(0, 1, len(GMM_OUTPUT))
plt.figure(1)
plt.plot(X, GMM_OUTPUT, label='GMM_OUTPUT')
plt.plot(X, DMM_OUTPUT, label='DMM_OUTPUT')
plt.legend(loc='upper left')
plt.show()
print(GMMradius)
print(DMMradius)
