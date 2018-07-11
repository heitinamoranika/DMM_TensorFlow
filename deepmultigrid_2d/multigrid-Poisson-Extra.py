import numpy as np
import tensorflow as tf
from basicequation import *
from multigrid import *
from optimizer import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# gridsize
inputsize = (2**5-1)**2
inputsize = int(inputsize)
sqrtinputsize = int(np.sqrt(inputsize))
outputsize = int((2**(np.log2(np.sqrt(inputsize)+1)-1)-1)**2)
sqrtoutputsize = int(np.sqrt(outputsize))
h = 1/(sqrtinputsize+1)


#form grid
[X, Y] = np.meshgrid(np.linspace(0, 1, sqrtinputsize),
                     np.linspace(0, 1, sqrtinputsize))
Coor = np.hstack((X.reshape([inputsize, 1]), Y.reshape([inputsize, 1])))
TriCoor = [[1, 0, sqrtinputsize], [1, sqrtinputsize+1, sqrtinputsize]]
TriCoor = np.kron(TriCoor, np.ones([sqrtinputsize-1, 1]))+np.kron(np.ones(
    [2, 3]), np.linspace(0, sqrtinputsize-2, sqrtinputsize-1).reshape([sqrtinputsize-1, 1]))
TriCoor = np.kron(TriCoor, np.ones([sqrtinputsize-1, 1]))+np.kron(np.ones([np.size(TriCoor, 0), np.size(TriCoor, 1)]), np.linspace(
    0, sqrtinputsize-2, sqrtinputsize-1).reshape([sqrtinputsize-1, 1]) * sqrtinputsize)


#Form Right F
RightF = np.zeros([inputsize, 1])
for i in range(np.size(TriCoor, 0)):
    N_Coor = TriCoor[i][:]
    Point = np.ones([3, 3])
    Point[0][1] = Coor[int(N_Coor[0]-1)][0]
    Point[1][1] = Coor[int(N_Coor[1]-1)][0]
    Point[2][1] = Coor[int(N_Coor[2]-1)][0]
    Point[0][2] = Coor[int(N_Coor[0]-1)][1]
    Point[1][2] = Coor[int(N_Coor[1]-1)][1]
    Point[2][2] = Coor[int(N_Coor[2]-1)][1]
    Triarea = 0.5*(1/(sqrtinputsize-1)**2)
    Inte = f((Point[0][1]+Point[1][1]+Point[2][1])/3,
             (Point[0][2]+Point[1][2]+Point[2][2])/3)*Triarea/3
    RightF[int(N_Coor[0])] = RightF[int(N_Coor[0])]+Inte
    RightF[int(N_Coor[1])] = RightF[int(N_Coor[1])]+Inte
    RightF[int(N_Coor[2])] = RightF[int(N_Coor[2])]+Inte

B = RightF.reshape([sqrtinputsize, sqrtinputsize])
B[0][:] = ub(np.linspace(h, 1-h, sqrtinputsize))
B[-1][:] = ut(np.linspace(h, 1-h, sqrtinputsize))
B[:][0] = ul(np.linspace(h, 1-h, sqrtinputsize))
B[:][-1] = ur(np.linspace(h, 1-h, sqrtinputsize))
B = B.reshape([inputsize, 1])


#from A
A0 = Possion(inputsize)
A1 = PossionExtra(inputsize)


#form R P
R = Restriction(inputsize)
P = Interpolation(inputsize)



#Linear
ULinear = np.linalg.solve(A1, B).reshape([sqrtinputsize, sqrtinputsize])
#Linear Output
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, ULinear, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#GMM
w = 2/3
smooth = 10
NUM_EPOCH = 100
UGMM, RESIDUALGMM = Multigrid_circle(
    inputsize, A1, B, P, R, smooth, w, NUM_EPOCH)
UGMM = UGMM.reshape([sqrtinputsize, sqrtinputsize])
#GMM Output
fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, UGMM, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#DMM
stepsize = 0.1
GMMradius, DMMradius, R0, P0, w0 = homotopyoptimizer(inputsize, smooth, A0, A1, P, R, w, stepsize)
UDMM, RESIDUALDMM = Multigrid_circle( inputsize, A1, B, P0, R0, smooth, w0, NUM_EPOCH)
UDMM = UDMM.reshape([sqrtinputsize, sqrtinputsize])
#DMM Output
fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, UDMM, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


print('GMMradius')
print(GMMradius)
print('GMMRESIDUAL')
print(RESIDUALGMM)

print('DMMradius')
print(DMMradius)
print('DMMRESIDUAL')
print(RESIDUALDMM)
[a, b] = P0.shape
import xlwt
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("P0")
for i in range(a):
    for j in range(b):
        sheet.write(i, j, P0[i][j])
workbook.save('T:\Programming\deepmultigrid_2d\P0.xls')

[a, b] = R0.shape
import xlwt
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("R0")
for i in range(a):
    for j in range(b):
        sheet.write(j, i, R0[i][j])
workbook.save('T:\Programming\deepmultigrid_2d\R0.xls')
