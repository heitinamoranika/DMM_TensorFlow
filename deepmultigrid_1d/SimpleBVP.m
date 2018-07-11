% Function to solve d^2ydx^2+y = 0.
function SimpleBVP()
 
solinit = bvpinit(linspace(0,1,50),[1 0]);
 
sol = bvp4c(@twoode,@twobc,solinit);
 
x = linspace(0,1);
y = deval(sol,x);
plot(x,y(1,:))
 
function dydx = twoode(x,y)
dydx = [ y(2); -abs(y(1)) ];
function res = twobc(ya,yb)
res = [ ya(1); yb(1) + 2 ];