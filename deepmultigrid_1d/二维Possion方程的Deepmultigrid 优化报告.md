# 二维Deep Multigrid 优化报告

姓名: 王恩泽

学号: 515071910069

班级: F1507104



[TOC]

# 对于优化过后的P, R

考虑优化过后的P和R.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from basicpossion import *
from spectralradius import *
from norm import *
from local_optimizer_Adam import *
from homotopy_optimizer_Helmholtz import *
from multigrid_two_level import *
```

```python
m = 8
kmax = 100
def f(x): return np.exp(np.pi*x)
```

```python
m, GMMradius, DMMradius, R0, P0, w0 = radiusPlot(m, 2, kmax, 5)
```

```python
print(GMMradius)
```

```
0.25074765958005774
```

```python
print(DMMradius)
```

```
0.19221522097271113
```

```python
[a,b] = P0.shape
print(a)
print(b)
import xlwt
workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("P0")
for i in range(a):
    for j in range(b):
        sheet.write(i,j,P0[i][j]) 
workbook.save('T:\Programming\deepmultigrid_1d\P0.xls')
```

```python
[a,b] = R0.shape
print(a)
print(b)
import xlwt
workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("R0")
for i in range(a):
    for j in range(b):
        sheet.write(j,i,R0[i][j]) 
workbook.save('T:\Programming\deepmultigrid_1d\R0.xls')
```

画图P

![](T:\Programming\deepmultigrid_1d\P0.jpg)

画图R

![](T:\Programming\deepmultigrid_1d\R0.jpg)

相较于正常的R, P, 优化过后的R, P带有一定的震荡部分, 但是对于这一震荡部分的性质我们不是很清楚.

把R0 第100行拿出来:

![](T:\Programming\deepmultigrid_1d\R0一行.png)

取快速傅里叶后在复平面上画图, 发现在半径为0.5的单位圆周围:

调整**kmax = 50**:

![](T:\Programming\deepmultigrid_1d\fftR0kmax=100.png)

**kmax = 100**

![](T:\Programming\deepmultigrid_1d\fftR0.png)

**kmax = 150**

![](T:\Programming\deepmultigrid_1d\fftR0kmax=150.png)

**kmax = 200**

![](T:\Programming\deepmultigrid_1d\fftR0kmax=200.png)

此时的P0第100行:

![](T:\Programming\deepmultigrid_1d\fftP0kmax=200.png)

是一个半径为1的单位圆.

没有经过优化的R0, P0:

![](T:\Programming\deepmultigrid_1d\fftR0without.png)

![](T:\Programming\deepmultigrid_1d\fftP0without.png)

所以优化的过程, 实际上就是让R0, P0进行微调, 来适应我们的问题的过程.



# 二维Poisson方程的优化

考虑二维Poisson方程:


$$
f(x, y) = x^2+y^2\\
 ub(x)= 0\\
 ut(x)= 0.5 x^2\\

ul(y)=\sin(\pi y) \\

 ur(y)=\exp(\pi)\sin(\pi y)+0.5y^2\\
$$

光滑次数取10次, Cycle Multigrid迭代次数取5次, 优化时的stepsize取0.1. 得到结果:

| Method |   Spectral Radius    |
| :----: | :------------------: |
|  GMM   | 0.02140858013307146  |
|  DMM   | 0.018724392905105254 |

GMM Method:

![](T:\Programming\deepmultigrid_1d\GMM1.png)

DMM Method:

![](T:\Programming\deepmultigrid_1d\DMM1.png)



可见两者还是相近的, 

同样的, 我们可以把P, R, 画出来:

P0:

![](T:\Programming\deepmultigrid_2d\P0.png)

R0:

![](T:\Programming\deepmultigrid_2d\R0.png)



实际上, 优化后的R, P 在山脊之外的值是非常小的, 相比于中间的山脊几乎是可以忽略的.



# Poisson–Boltzmann equation的优化

我们考虑以下PDE:
$$
-\nabla(a(x)\nabla u(x)) = f(x)
$$
相对于Poisson-Boltzmann方程, 这个方程式线性的, 所以也属于Poisson-Boltzmann方程的一种特殊类型.

显然它的有限元的双线性泛函可以写成:
$$
\iint a(x)\dfrac{\partial \phi_i}{\partial x}\dfrac{\partial \phi_j}{\partial x}+a(x) \dfrac{\partial \phi_i}{\partial y}\dfrac{\partial \phi_j}{\partial y}
$$
所以相对于Poisson方程的左端矩阵来说, 这个方程的左端矩阵A 是很方便写出来的.



# 初次试验

我们考虑以下方程:

```python
e = 1
def a(x, y): return np.sin(x/e)*np.sin(y/e)+3
def f(x, y): return (2*np.sin(2*np.pi*x) - 16*x*np.pi**2*np.sin(4*np.pi*y)*(x - 1))*(np.sin(x/e)*np.sin(y/e) + 3) + (2*np.sin(4*np.pi*y) - 4*y*np.pi**2*np.sin(2*np.pi*x)*(y - 1))*(np.sin(x/e)*np.sin(y/e) + 3) + (np.cos(y/e) * np.sin(x/e)*(y*np.sin(2*np.pi*x) + np.sin(2*np.pi*x)*(y - 1) + 4*x*np.pi*np.cos(4*np.pi*y)*(x - 1)))/e + (np.sin(y/e)*np.cos(x/e)*x*np.sin(4*np.pi*y) + np.sin(4*np.pi*y)*(x - 1) + *y*np.pi*np.cos(2*np.pi*x)*(y - 1)))/e
def ub(x): return 0
def ut(x): return 0
def ul(y): return 0
def ur(y): return 0
def ureal(x, y): return x*(1-x)*np.sin(4*np.pi*y)+y*(1-y)*np.sin(2*np.pi*x)
```

其中 $a(x,y)$ 是隐藏在算子中的震动函数, $e$ 是其中可调整的系数,  我们取$e = 1, 0.1, 0.01$画张图看看是什么:

|  e   |           sin(x/e)sin(y/e)+3           |
| :--: | :------------------------------------: |
|  1   |  ![](T:\NumericalHomework\HW6\e1.png)  |
| 0.1  | ![](T:\NumericalHomework\HW6\e01.png)  |
| 0.01 | ![](T:\NumericalHomework\HW6\e001.png) |



可见随着 $e$ 越来越靠近0, 震动函数震动的越剧烈, 基于对1维霍姆霍兹方程的考量, 我的猜想是当 $e$小到一定程度的时候, 这个震动最终会干扰到数值解法我们主要研究的是接近0量级发生的事情, 解析解是$ureal(x,y)$,  $f(x,y)$ 是重新计算后的右端函数, 我们看一下结果:

##e=1

 解析解:

![](T:\Programming\deepmultigrid_2d\real2.png)

Linear:

![](T:\Programming\deepmultigrid_2d\linear2.png)

GMM:

![](T:\Programming\deepmultigrid_2d\GMM2.png)

DMM:

![](T:\Programming\deepmultigrid_2d\GMM2.png)

| Method |        Radius        |
| :----: | :------------------: |
|  GMM   | 0.007518778375142891 |
|  DMM   | 0.007518778375142891 |

可见两者几乎是一样的, 考察Residual下降的次数:
![](T:\Programming\deepmultigrid_2d\e1.PNG)



##e=0.01

GMM:

![](T:\Programming\deepmultigrid_2d\GMM001.png)

![](T:\Programming\deepmultigrid_2d\DMM001.png)



| Method |        Radius        |
| :----: | :------------------: |
|  GMM   | 0.008347313912932287 |
|  DMM   | 0.008347313912932284 |



## e = 0.001

Linear Method:

![](T:\Programming\deepmultigrid_2d\GS0001.png)

GMM:

![](T:\Programming\deepmultigrid_2d\GMM0001.png)

DMM:

![](T:\Programming\deepmultigrid_2d\DMM0001.png)

可见, 和一维情况类似, 我们只能让数值方法变快, 而不能让他变好.

而且优化的结果也不好, 对于Helmholtz方程来说, 即使我们的步长只取0.1, 谱半径经过优化还是不断下降的, 但是这样的方程却不会.





# 代码汇总

