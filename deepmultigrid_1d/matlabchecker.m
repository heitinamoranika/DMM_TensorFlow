m = 7;
for i =1:100
    P = interpolate1d(m);
    R = restriction1d(m);
    M = ((2/3)^(-1)).*diag(diag(Helmholtz(m, Kmax(i))));
    K= M- Helmholtz(m, Kmax(i));
    Kmax = linspace(0,500,100);
    Radius(i) = max(abs(eig((inv(M)*K)*(eye(2^(m+1)-1)+P*inv(R*Helmholtz(m, Kmax(i))*P)*R*Helmholtz(m, Kmax(i)))*(inv(M)*K))));
end
plot(Kmax, Radius, 'g');
hold on

m = 6;
for i =1:100
    P = interpolate1d(m);
    R = restriction1d(m);
    M = ((2/3)^(-1)).*diag(diag(Helmholtz(m, Kmax(i))));
    K= M- Helmholtz(m, Kmax(i));
    Kmax = linspace(0,500,100);
    Radius(i) = max(abs(eig((inv(M)*K)*(eye(2^(m+1)-1)+P*inv(R*Helmholtz(m, Kmax(i))*P)*R*Helmholtz(m, Kmax(i)))*(inv(M)*K))));
end
plot(Kmax, Radius, 'y');
hold on

m = 5;
for i =1:100
    P = interpolate1d(m);
    R = restriction1d(m);
    M = ((2/3)^(-1)).*diag(diag(Helmholtz(m, Kmax(i))));
    K= M- Helmholtz(m, Kmax(i));
    Kmax = linspace(0,500,100);
    Radius(i) = max(abs(eig((inv(M)*K)*(eye(2^(m+1)-1)+P*inv(R*Helmholtz(m, Kmax(i))*P)*R*Helmholtz(m, Kmax(i)))*(inv(M)*K))));
end
plot(Kmax, Radius, 'r');
hold on

m = 4;
for i =1:100
    P = interpolate1d(m);
    R = restriction1d(m);
    M = ((2/3)^(-1)).*diag(diag(Helmholtz(m, Kmax(i))));
    K= M- Helmholtz(m, Kmax(i));
    Kmax = linspace(0,500,100);
    Radius(i) = max(abs(eig((inv(M)*K)*(eye(2^(m+1)-1)+P*inv(R*Helmholtz(m, Kmax(i))*P)*R*Helmholtz(m, Kmax(i)))*(inv(M)*K))));
end
plot(Kmax, Radius, 'k');
hold on

m = 3;
for i =1:100
    P = interpolate1d(m);
    R = restriction1d(m);
    M = ((2/3)^(-1)).*diag(diag(Helmholtz(m, Kmax(i))));
    K= M- Helmholtz(m, Kmax(i));
    Kmax = linspace(0,500,100);
    Radius(i) = max(abs(eig((inv(M)*K)*(eye(2^(m+1)-1)+P*inv(R*Helmholtz(m, Kmax(i))*P)*R*Helmholtz(m, Kmax(i)))*(inv(M)*K))));
end
plot(Kmax, Radius, 'b');
hold on



ylim([0,10]);