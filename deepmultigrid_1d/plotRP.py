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
kmax = 100
def f(x): return np.exp(np.pi*x)
m, GMMradius, DMMradius, R0, P0, w0 = radiusPlot(m, 2, kmax, 5)
X,Y = P0.shape
X, Y = np.meshgrid(X, Y)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, P0, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()