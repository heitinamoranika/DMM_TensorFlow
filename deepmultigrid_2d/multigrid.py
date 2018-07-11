import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicequation import*


def Multigrid_two_grid(inputsize, A, B, P, R, s, w, U0):
    M = np.matrix(w**(-1)*np.diag(np.diag(A)))
    K = M - A
    C = np.linalg.inv(M)*K
    b = np.linalg.inv(M)*B
    for i in range(s):
        U0 = C*U0+b
    r = B - A*U0
    Residual = np.linalg.norm(r,2)
    rc = R*r
    Ac = R*A*P
    Uc = np.linalg.solve(Ac, rc)
    U = U0 + P*Uc
    for i in range(s):
        U = C*U+b
    return U, Residual


def Multigrid_circle(inputsize, A, B, P, R, s, w, NUM_EPOCH):
    U0 = np.matrix(np.zeros([inputsize, 1]))
    RESIDUAL = []
    for i in range(NUM_EPOCH):
        U0, Residual = Multigrid_two_grid(inputsize, A, B, P, R, s, w, U0)
        RESIDUAL.append(Residual)
    return U0, RESIDUAL
