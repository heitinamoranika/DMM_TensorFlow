import numpy as np
import tensorflow as tf

from basicequation import *


def NormNumpy(inputsize, A, P, R, w, smooth):
    M = (w**(-1)) * np.diag(np.diag(A))
    K = M - A
    MK = np.linalg.inv(M)*K
    I = np.matrix(np.eye(inputsize))
    IPRAPRA = I - P*np.linalg.inv(R*A*P)*R*A
    C = (MK**smooth)*IPRAPRA*(MK**smooth)
    radius = np.linalg.norm(C, 2)
    return radius


def Local_optimizer_Adam(inputsize, A0, P0, R0, w0, smooth):
    radiusold = NormNumpy(inputsize, A0, P0, R0, w0, smooth)
    A = tf.constant(A0, dtype=tf.float64)
    P = tf.Variable(initial_value=P0, dtype=tf.float64)
    R = tf.Variable(initial_value=R0, dtype=tf.float64)
    w = tf.Variable(initial_value=w0, dtype=tf.float64)
    M10 = tf.constant(np.diag(np.diag(A0)), dtype=tf.float64)
    I = tf.constant(np.eye(inputsize), dtype=tf.float64)
    M = (w**(-1)) * M10
    K = M - A
    RAP = tf.matrix_inverse(tf.matmul(R, tf.matmul(A, P)))
    IPRAPRA = I - tf.matmul(P, tf.matmul(RAP, tf.matmul(R, A)))
    MK = tf.pow(tf.matmul(tf.matrix_inverse(M), K), smooth)
    loss1 = tf.norm(MK*IPRAPRA*MK, ord=2)
    optimizer = tf.train.AdamOptimizer()
    train_op1 = optimizer.minimize(loss1)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(200):
        sess.run(train_op1)
    R = sess.run(R)
    P = sess.run(P)
    w = sess.run(w)
    radiusnew = NormNumpy(inputsize, A0, P, R, w, smooth)
    return R, P, w, radiusold, radiusnew


def homotopyoptimizer(inputsize, smooth, A0, A1, P0, R0, w0, step):
    M = A0
    GMMradius = NormNumpy(inputsize, A1, P0, R0, w0, smooth)
    print(GMMradius)
    accept_radius = GMMradius
    L = step
    while L < 1:
        Rnew, Pnew, wnew, radiusold, radiusnew = Local_optimizer_Adam(
            inputsize, M, P0, R0, w0, smooth)
        R0 = Rnew
        P0 = Pnew
        w0 = wnew
        L = L + step
        print(radiusnew)
        print(L)
        M = M - step * A0 + step * A1
    DMMradius = NormNumpy(inputsize, A1, P0, R0, w0, smooth)
    return GMMradius, DMMradius, R0, P0, w0
