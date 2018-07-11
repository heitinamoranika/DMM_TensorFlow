import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from basicpossion import *
from spectralradius import *
from norm import *
from local_optimizer_Adam import *
from homotopy_optimizer_Possion import *
from multigrid_two_level import *

m = 8
s = 2
R0, P0, w, GMMradius, DMMradius = Local_optimizer_Adam(m,Possion(m), interpolate(m), restriction(m), 2/3, s, 20)
Raw_Restriction = restriction(m)
Raw_Interpolate = interpolate(m)
Optimized_Restriction = R0
Optimized_Interpolate = P0


print('----------------------------------Raw_Restriction----------------------------------')
n = Raw_Restriction.shape[1];
for i in range(n):
    X = np.zeros([n,1]);
    X[i] = 1;
    OUTPUT = np.dot(Raw_Restriction,X);
    U = np.linspace(0,1,len(OUTPUT));
    print('X is');
    print(X);
    print('OUTPUT is');
    print(OUTPUT);
    plt.figure(1)
    plt.plot(U, OUTPUT,label='Raw_Restriction');
    plt.xlabel('u')
    plt.ylabel('OUTPUT')
    plt.legend(loc='upper left')
    plt.show()

print('----------------------------------Raw_Interpolate----------------------------------')
n = Raw_Interpolate.shape[1];
for i in range(n):
    X = np.zeros([n,1]);
    X[i] = 1;
    OUTPUT = np.dot(Raw_Interpolate,X);
    U = np.linspace(0,1,len(OUTPUT));
    print('X is');
    print(X);
    print('OUTPUT is');
    print(OUTPUT);
    plt.figure(1)
    plt.plot(U, OUTPUT,label='Raw_Interpolate');
    plt.xlabel('u')
    plt.ylabel('OUTPUT')
    plt.legend(loc='upper left')
    plt.show()

print('----------------------------------Optimized_Restriction----------------------------------')
n = Optimized_Restriction.shape[1];
for i in range(n):
    X = np.zeros([n,1]);
    X[i] = 1;
    OUTPUT = np.dot(Optimized_Restriction,X);
    U = np.linspace(0,1,len(OUTPUT));
    print('X is');
    print(X);
    print('OUTPUT is');
    print(OUTPUT);
    plt.figure(1)
    plt.plot(U, OUTPUT,label='Optimized_Restriction');
    plt.xlabel('u')
    plt.ylabel('OUTPUT')
    plt.legend(loc='upper left')
    plt.show()

print('----------------------------------Optimized_Interpolate----------------------------------')
n = Optimized_Interpolate.shape[1];
for i in range(n):
    X = np.zeros([n,1]);
    X[i] = 1;
    OUTPUT = np.dot(Optimized_Interpolate,X);
    U = np.linspace(0,1,len(OUTPUT));
    print('X is');
    print(X);
    print('OUTPUT is');
    print(OUTPUT);
    plt.figure(1)
    plt.plot(U, OUTPUT,label='Optimized_Interpolate');
    plt.xlabel('u')
    plt.ylabel('OUTPUT')
    plt.legend(loc='upper left')
    plt.show()