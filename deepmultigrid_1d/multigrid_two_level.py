import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import*


def Multigrid_two_level(m, A, P, R, s, w, f, U0):
    F = np.zeros([2**(m+1)-1, 1])
    for i in range(2**(m+1)-1):
            F[i] = f(i/(2**(m+1)-1))
    M = w**(-1)*np.diag(np.diag(A))
    K = M - A
    for i in range(s):
        U0 = np.dot(np.dot(np.linalg.inv(M), K), U0) + \
            np.dot(np.linalg.inv(M), F)
    r = F - np.dot(A, U0)
    rc = np.dot(R, r)
    Ac = np.dot(R, np.dot(A, P))
    Uc = np.dot(np.linalg.inv(Ac), rc)
    U = U0 + np.dot(P, Uc)
    for i in range(s):
        U = np.dot(np.dot(np.linalg.inv(M), K), U) + \
            np.dot(np.linalg.inv(M), F)
    return U


def Multigrid_circle(m, A, P, R, s, w, f, NUM_EPOCH):
    U0 = np.matrix(np.random.uniform(-0.1, 0.1, size=[2**(m+1)-1, 1]))
    for i in range(NUM_EPOCH):
        U0 = Multigrid_two_level(m, A, P, R, s, w, f, U0)
    return U0