import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import *
from spectralradius import *
from norm import *
from local_optimizer_Adam import *


def radiusPlot(m, s):
    NUM_EPOCHS = 5
    step = 0.1
    M = Possion(m)
    R0 = restriction(m)
    P0 = interpolate(m)
    w0 = 2 / 3
    GMMradius = RadiusNumpy(m, M, P0, R0, w0, s)
    print(GMMradius)
    accept_radius = 0.5
    L = step
    while L < 1:
        Rnew, Pnew, wnew, radiusold, radiusnew = Local_optimizer_Adam(
            m, M, P0, R0, w0, s, NUM_EPOCHS)
        if radiusnew < accept_radius:
            R0 = Rnew
            P0 = Pnew
            w0 = wnew
            L = L + step
            print(radiusnew)
            print(L)
        else:
            step = step * 0.1
            print('OMG_Fail!!')
    DMMradius = RadiusNumpy(m, M, P0, R0, w0, s)
    return m, GMMradius, DMMradius, R0, P0, w0