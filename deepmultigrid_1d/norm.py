import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def NormNumpy(m, A, P, R, w, s):
    M = (w**(-1)) * np.diag(np.diag(A))
    K = M - A
    MK = np.matmul(np.linalg.inv(M), K)
    IPRAPRA = np.eye(2**(m + 1) - 1) + np.matmul(np.matmul
                                                 (np.matmul(P, np.linalg.inv(np.matmul(np.matmul(R, A), P))), R), A)
    C = np.matmul(MK**s, np.matmul(IPRAPRA, MK**s))
    radius = np.linalg.norm(C,2)
    return radius
