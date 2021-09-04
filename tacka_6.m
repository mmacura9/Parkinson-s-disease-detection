close all
clear
clc
f = fopen('parkinsons.data');
t = textscan(f, '%s');
fclose(f);

matrix = split(t{1},',');
var = str2double(matrix(2:end,2:end));
data = var(:,[1,3,5,9,13,18,19,20,21,22, 17]);

p0 = sum(data(:,end)==0);
p1 = sum(data(:,end)==1);

M = p0+p1;

p1 = p1/M;
p0 = p0/M;

X0 = data(data(:,end) == 0, 1:end-1);
X1 = data(data(:,end) == 1, 1:end-1);

S0 = cov(X0);
S1 = cov(X1);

Sw = p0*S0 + p1*S1;

M0 = mean(X0);
M1 = mean(X1);

M = p0*M0 + p1*M1;
Sb = p0*(M0-M)'*(M0-M)+p1*(M1-M)'*(M1-M);

Sm = Sw+Sb;
S = Sw^-1*Sb;
J = sum(diag(S));

[V,D] = eig(S);
D = real(D);
D = diag(D);
A = real([V(:,1) V(:,2)]);

Y0 = A'*X0';
Y1 = A'*X1';

figure
    plot(Y0(1,:),Y0(2,:),'o')
    hold all
    plot(Y1(1,:),Y1(2,:),'c*')








