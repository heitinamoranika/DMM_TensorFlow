clc
clear
syms x y e 
a = @(x,y,e) sin(x/e)*sin(y/e)+3;
u = @(x,y) x*(1-x)*sin(4*pi*y)+y*(1-y)*sin(2*pi*x);
f = simplify(-diff(a*diff(u,x),x) - diff(a*diff(u,y),y))
