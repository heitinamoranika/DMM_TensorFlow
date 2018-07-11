import numpy as np
from scipy.integrate import dblquad

e1 = 1/5
e2 = 1/13
e3 = 1/17
e4 = 1/31
e5 = 1/65


def ub(x): return 0
def ut(x): return 0
def ul(y): return 0
def ur(y): return 0
def a(x, y): return (1/6)*((1.1+np.sin(2*np.pi*x/e1))/(1.1+np.sin(2*np.pi*y/e1))+(1.1+np.sin(2*np.pi*y/e2))/(1.1+np.cos(2*np.pi*x/e2))+(1.1+np.cos(2*np.pi*x/e3)) /
                           (1.1+np.sin(2*np.pi*y/e3))+(1.1+np.sin(2*np.pi*y/e4))/(1.1+np.cos(2*np.pi*x/e4))+(1.1+np.cos(2*np.pi*x/e5))/(1.1+np.sin(2*np.pi*y/e5))+np.sin(4*(x**2)*(y**2))+1)


def f(x, y): return (x*np.sin(4*np.pi*y) + np.sin(4*np.pi*y)*(x - 1) + 2*y*np.pi*np.cos(2*np.pi*x)*(y - 1))*((4*x*y**2*np.cos(4*x**2*y**2))/3 + (5*np.pi*np.cos(10*np.pi*x))/(3*(np.sin(10*np.pi*y) + 11/10)) - (17*np.pi*np.sin(34*np.pi*x))/(3*(np.sin(34*np.pi*y) + 11/10)) - (65*np.pi*np.sin(130*np.pi*x))/(3*(np.sin(130*np.pi*y) + 11/10)) + (13*np.pi*np.sin(26*np.pi*x)*(np.sin(26*np.pi*y) + 11/10))/(3*(np.cos(26*x*np.pi) + 11/10)**2) + (31*np.pi*np.sin(62*np.pi*x)*(np.sin(62*np.pi*y) + 11/10))/(3*(np.cos(62*x*np.pi) + 11/10)**2)) + (2*np.sin(2*np.pi*x) - 16*x*np.pi**2*np.sin(4*np.pi*y)*(x - 1))*(np.sin(4*x**2*y**2)/6 + (np.sin(26*np.pi*y) + 11/10)/(6*(np.cos(26*np.pi*x) + 11/10)) + (np.cos(34*np.pi*x) + 11/10)/(6*(np.sin(34*np.pi*y) + 11/10)) + (np.sin(62*np.pi*y) + 11/10)/(6*(np.cos(62*np.pi*x) + 11/10)) + (np.cos(130*np.pi*x) + 11/10)/(6*(np.sin(130*np.pi*y) + 11/10)) + (np.sin(10*np.pi*x) + 11/10)/(6*(np.sin(10*np.pi*y) + 11/10)) + 1/6) + \
               (2*np.sin(4*np.pi*y) - 4*y*np.pi**2*np.sin(2*np.pi*x)*(y - 1))*(np.sin(4*x**2*y**2)/6 + (np.sin(26*np.pi*y) + 11/10)/(6*(np.cos(26*np.pi*x) + 11/10)) + (np.cos(34*np.pi*x) + 11/10)/(6*(np.sin(34*np.pi*y) + 11/10)) + (np.sin(62*np.pi*y) + 11/10)/(6*(np.cos(62*np.pi*x) + 11/10)) + (np.cos(130*np.pi*x) + 11/10)/(6*(np.sin(130*np.pi*y) + 11/10)) + (np.sin(10*np.pi*x) + 11/10)/(6*(np.sin(10*np.pi*y) + 11/10)) + 1/6) + (y*np.sin(2*np.pi*x) + np.sin(2*np.pi*x)*(y - 1) + 4*x*np.pi *
                                                                                                                                                                                                                                                                                                                                                                                                                                      np.cos(4*np.pi*y)*(x - 1))*((4*x**2*y*np.cos(4*x**2*y**2))/3 + (13*np.pi*np.cos(26*np.pi*y))/(3*(np.cos(26*np.pi*x) + 11/10)) + (31*np.pi*np.cos(62*np.pi*y))/(3*(np.cos(62*np.pi*x) + 11/10)) - (17*np.pi*np.cos(34*np.pi*y)*(np.cos(34*np.pi*x) + 11/10))/(3*(np.sin(34*y*np.pi) + 11/10)**2) - (65*np.pi*np.cos(130*np.pi*y)*(np.cos(130*np.pi*x) + 11/10))/(3*(np.sin(130*y*np.pi) + 11/10)**2) - (5*np.pi*np.cos(10*np.pi*y)*(np.sin(10*np.pi*x) + 11/10))/(3*(np.sin(10*y*np.pi) + 11/10)**2))
def ureal(x, y): x*(1-x)*np.sin(4*np.pi*y)+y*(1-y)*np.sin(2*np.pi*x)


'''
e = 0.001
def a(x, y): return np.sin(x/e)*np.sin(y/e)+3
def f(x, y): return (2*np.sin(2*np.pi*x) - 16*x*np.pi**2*np.sin(4*np.pi*y)*(x - 1))*(np.sin(x/e)*np.sin(y/e) + 3) + (2*np.sin(4*np.pi*y) - 4*y*np.pi**2*np.sin(2*np.pi*x)*(y - 1))*(np.sin(x/e)*np.sin(y/e) + 3) + (np.cos(y/e)                                                                                                                                                                                                                    * np.sin(x/e)*(y*np.sin(2*np.pi*x) + np.sin(2*np.pi*x)*(y - 1) + 4*x*np.pi*np.cos(4*np.pi*y)*(x - 1)))/e + (np.sin(y/e)*np.cos(x/e)*(x*np.sin(4*np.pi*y) + np.sin(4*np.pi*y)*(x - 1) + 2*y*np.pi*np.cos(2*np.pi*x)*(y - 1)))/e
def ub(x): return 0
def ut(x): return 0
def ul(y): return 0
def ur(y): return 0
def ureal(x, y): return x*(1-x)*np.sin(4*np.pi*y)+y*(1-y)*np.sin(2*np.pi*x)
'''
def Restriction(inputsize):
    inputsize = int(inputsize)
    sqrtinputsize = int(np.sqrt(inputsize))
    outputsize = int((2**(np.log2(np.sqrt(inputsize)+1)-1)-1)**2)
    sqrtoutputsize = int(np.sqrt(outputsize))
    OUTPUT = np.zeros([outputsize, inputsize])
    for j in range(sqrtoutputsize):
        for i in range(sqrtoutputsize):
            OUTPUT[i+sqrtoutputsize*j][2*i+j*2*sqrtinputsize] = 1/16
            OUTPUT[i+sqrtoutputsize*j][1+2*i+j*2*sqrtinputsize] = 1/8
            OUTPUT[i+sqrtoutputsize*j][2+2*i+j*2*sqrtinputsize] = 1/16
            OUTPUT[i+sqrtoutputsize*j][sqrtinputsize +
                                       2*i+j*2*sqrtinputsize] = 1/8
            OUTPUT[i+sqrtoutputsize*j][1+sqrtinputsize +
                                       2*i+j*2*sqrtinputsize] = 1/4
            OUTPUT[i+sqrtoutputsize*j][2+sqrtinputsize +
                                       2*i+j*2*sqrtinputsize] = 1/8
            OUTPUT[i+sqrtoutputsize*j][2*sqrtinputsize +
                                       2*i+j*2*sqrtinputsize] = 1/16
            OUTPUT[i+sqrtoutputsize*j][1+2 *
                                       sqrtinputsize + 2*i+j*2*sqrtinputsize] = 1/8
            OUTPUT[i+sqrtoutputsize*j][2+2 *
                                       sqrtinputsize+2*i+j*2*sqrtinputsize] = 1/16
    return np.matrix(OUTPUT)


def Interpolation(inputsize):
    inputsize = int(inputsize)
    OUTPUT = 4*np.transpose(Restriction(inputsize))
    return np.matrix(OUTPUT)


def Possion(inputsize):
    inputsize = int(inputsize)
    sqrtinputsize = int(np.sqrt(inputsize))
    A1 = 2*np.eye(sqrtinputsize)
    for i in range(sqrtinputsize-1):
        A1[i, i+1] = -1
        A1[i+1, i] = -1
    I = np.eye(sqrtinputsize)
    OUTPUT = (np.kron(I, A1)+np.kron(A1, I))
    return np.matrix(OUTPUT)


def PossionExtra(inputsize):
    inputsize = int(inputsize)
    sqrtinputsize = int(np.sqrt(inputsize))
    area = 1
    [X, Y] = np.meshgrid(np.linspace(0, 1, sqrtinputsize),
                         np.linspace(0, 1, sqrtinputsize))
    A = area*(a(X, Y).reshape([inputsize, 1]))
    Poi = Possion(inputsize)
    for i in range(inputsize):
        Poi[i][:] = A[i]*Poi[i][:]
    OUTPUT = Poi
    return np.matrix(OUTPUT)
