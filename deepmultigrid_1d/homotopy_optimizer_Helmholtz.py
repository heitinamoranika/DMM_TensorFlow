import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import *
from spectralradius import *
from norm import *
from local_optimizer_Adam import *


def radiusPlot(m, s, kmax, NUM_EPOCHS):
    step = 0.1
    A0 = Possion(m)
    A1 = Helmholtz(m, kmax)
    R0 = restriction(m)
    P0 = interpolate(m)
    w0 = 2 / 3
    M = A0
    GMMradius = RadiusNumpy(m, A1, P0, R0, w0, s)
    print(GMMradius)
    accept_radius = 0.5
    L = step
    while L < 1:
        Rnew, Pnew, wnew, radiusold, radiusnew = Local_optimizer_Adam(m, M, P0, R0, w0, s, NUM_EPOCHS)
        if radiusnew < accept_radius:
            R0 = Rnew
            P0 = Pnew
            w0 = wnew
            L = L + step
            print(radiusnew)
            print(L)
            M = M - step * A0 + step * A1
        else:
            M = M + 2 * step * A0 - 2 * step * A1
            step = step * 0.1
            print('OMG_Fail!!')
    DMMradius = RadiusNumpy(m, A1, P0, R0, w0, s)
    return m, GMMradius, DMMradius, R0, P0, w0
