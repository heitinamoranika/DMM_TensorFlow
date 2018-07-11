import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def interpolate(m):
    INT = np.zeros([2**(m + 1) - 1, 2**m - 1])
    for i in range(2**m - 1):
        INT[2 * i, i] = 1 / 2
        INT[2 * i + 1, i] = 1
        INT[2 * i + 2, i] = 1 / 2
    return INT


def restriction(m):
    RESTRICT = np.zeros([2**m - 1, 2**(m + 1) - 1])
    for i in range(2**m - 1):
        RESTRICT[i, 2 * i] = 1 / 4
        RESTRICT[i, 2 * i + 1] = 1 / 2
        RESTRICT[i, 2 * i + 2] = 1 / 4
    return RESTRICT


def Possion(m):
    A = (2**(2 * m + 3)) * np.eye(2**(m + 1) - 1)
    for i in range(2**(m + 1) - 2):
        A[i, i + 1] = -(2**(2 * m + 2))
        A[i + 1, i] = -(2**(2 * m + 2))
    return A


def k(x, kmax):
    if x < 0.5:
        y = 1
    else:
        y = kmax
    return y


def Helmholtz(m, kmax):
    H = (2**(2 * m + 3)) * np.eye(2**(m + 1) - 1)
    for i in range(2**(m + 1) - 2):
        H[i, i + 1] = -(2**(2 * m + 2))
        H[i + 1, i] = -(2**(2 * m + 2))
    for i in range(2**(m + 1) - 1):
        H[i, i] = H[i, i] - k(i/(2**(m + 1) - 1), kmax)**2
    return H
