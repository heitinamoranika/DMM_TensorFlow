import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import *
from spectralradius import *
from norm import *


def Local_optimizer_Adam(m, A0, P0, R0, w0, s, NUM_EPOCHS):
    radiusold = RadiusNumpy(m, A0, P0, R0, w0, s)
    A = tf.constant(A0, dtype=tf.float64)
    P = tf.Variable(initial_value=P0, dtype=tf.float64)
    R = tf.Variable(initial_value=R0, dtype=tf.float64)
    w = tf.Variable(initial_value=w0, dtype=tf.float64)
    M10 = tf.constant(np.diag(np.diag(A0)), dtype=tf.float64)
    I = tf.constant(np.eye(2**(m + 1) - 1), dtype=tf.float64)
    M = w**(-1) * M10
    K = M - A
    RAP = tf.matrix_inverse(tf.matmul(R, tf.matmul(A, P)))
    IPRAPRA = I + tf.matmul(P, tf.matmul(RAP, tf.matmul(R, A)))
    MK = tf.pow(tf.matmul(tf.matrix_inverse(M), K), s)
    C = tf.matmul(MK, tf.matmul(IPRAPRA, MK))
    Cnorm = tf.norm(C, ord=2)
    loss = Cnorm
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    Norm = []
    for i in range(NUM_EPOCHS):
        sess.run(train_op)
    R = sess.run(R)
    P = sess.run(P)
    w = sess.run(w)
    radiusnew = RadiusNumpy(m, A0, P, R, w, s)
    return R, P, w, radiusold, radiusnew
