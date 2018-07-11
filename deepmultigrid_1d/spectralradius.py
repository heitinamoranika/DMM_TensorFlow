import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def RadiusNumpy(m, A, P, R, w, s):
    M = (w**(-1)) * np.diag(np.diag(A))
    K = M - A
    MK = np.matmul(np.linalg.inv(M), K)
    IPRAPRA = np.eye(2**(m + 1) - 1) + np.matmul(np.matmul
                                                 (np.matmul(P, np.linalg.inv(np.matmul(np.matmul(R, A), P))), R), A)
    C = np.matmul(MK**s, np.matmul(IPRAPRA, MK**s))
    e, v = np.linalg.eig(C)
    radius = max(abs(e))
    return radius


def RadiusTen(m, A0, P0, R0, w0, s):
    A = tf.constant(A0, dtype=tf.float64)
    P = tf.constant(P0, dtype=tf.float64)
    R = tf.constant(R0, dtype=tf.float64)
    w = tf.constant(w0, dtype=tf.float64)
    M10 = tf.constant(np.diag(np.diag(A0)), dtype=tf.float64)
    I = tf.constant(np.eye(2**(m + 1) - 1), dtype=tf.float64)
    M = (w**(-1)) * M10
    K = M - A
    RAP = tf.matrix_inverse(tf.matmul(R, tf.matmul(A, P)))
    IPRAPRA = I + tf.matmul(P, tf.matmul(RAP, tf.matmul(R ,A)))
    MK = tf.pow(tf.matmul(tf.matrix_inverse(M), K), s)
    C = tf.matmul(MK, tf.matmul(IPRAPRA, MK))
    e, v = tf.self_adjoint_eig(C, name="eigendata")
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    radius = max(abs(sess.run(e)))
    return radius
