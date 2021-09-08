%% Marko
close all
clear
clc
f = fopen('parkinsons.data');
t = textscan(f, '%s');
fclose(f);

matrix = split(t{1},',');
var = str2double(matrix(2:end,2:end));
data = var(:,[1,3,5,9,13,18,19,20,21,22, 17]);

%% Kristina
close all
clear
clc
matrix = table2array(readtable('parkinsonsData.csv'));
data = matrix(:,[1,3,5,9,13,18,19,20,21,22, 17]);

%% Normalizacija
[M N] = size(data);
for i = 1:N
    data(:,i)= data(:,i)/max(data(:,i));
end
%% Podela
% holdout metoda (80% obucavanje, 20% testiranje)
r = 4:4:length(data);
data_test = data(r,:);

l = 1:length(data);
l = l(mod(l(:),4) ~= 0);

data_trening = data(l,:);

%% Nasumicna podela
X1 = data(data(:,end) == 0, 1:end-1);
X2 = data(data(:,end) == 1, 1:end-1);

[m1,~] = size(X1);
[m2,~] = size(X2);

r = 0.7;
% Obucavajuci i testirajuci za klasu 1
tmp = randperm(m1);
ind_ob = tmp(1:round(r*m1));
X1_ob = X1(ind_ob,:);
ind_t = tmp(round(r*m1)+1:end);
X1_t = X1(ind_t,:);

% Obucavajuci i testirajuci za klasu 2
tmp = randperm(m2);
ind_ob = tmp(1:round(r*m2));
X2_ob = X2(ind_ob,:);
ind_t = tmp(round(r*m2)+1:end);
X2_t = X2(ind_t,:);

%% Balansiranje

diff_ob = size(X2_ob,1) - size(X1_ob,1);
diff_t = size(X2_t,1) - size(X1_t,1);

M_ob = mean(X1_ob, 1);
S_ob = cov(X1_ob, 1);
new_ob = mvnrnd(M_ob, S_ob, diff_ob);

X1_ob = [X1_ob; new_ob];

%% spajanje

[dim1,~] = size(X1_ob);
[dim2,~] = size(X2_ob);
X_ob = [X1_ob; X2_ob];
y_ob = [zeros(1,dim1) ones(1,dim2)]';

[dim1,~] = size(X1_t);
[dim2,~] = size(X2_t);
X_t = [X1_t; X2_t];
y_t = [zeros(1,dim1) ones(1,dim2)]';

data_trening = [X_ob, y_ob(:,1)];
data_test = [X_t, y_t(:,1)];

%% LDA
p0 = sum(data_trening(:,end)==0);
p1 = sum(data_trening(:,end)==1);

M = p0+p1;

p1 = p1/M;
p0 = p0/M;

X0 = data_trening(data_trening(:,end) == 0, 1:end-1);
X1 = data_trening(data_trening(:,end) == 1, 1:end-1);

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

[~, ind] = sort(D, 'descend');
A = real([V(:,ind(1)) V(:,ind(2))]);

Y0 = A'*X0';
Y1 = A'*X1';

figure(1)
    plot(Y0(1,:),Y0(2,:),'o')
    hold all
    plot(Y1(1,:),Y1(2,:),'c*')
    title('LDA redukcija na 2 dimenzije')
    legend('zdravi', 'bolesni')
    hold off
% %% 3 dimenzije
% A = real([V(:,ind(1)) V(:,ind(2)) V(:,ind(3))]);
% 
% Y0 = A'*X0';
% Y1 = A'*X1';
% 
% figure(2)
%     plot3(Y0(1,:),Y0(2,:),Y0(3,:),'o')
%     hold all
%     plot3(Y1(1,:),Y1(2,:),Y1(3,:),'c*')
%     title('LDA redukcija na 3 dimenzije')
%     legend('zdravi', 'bolesni')
%     hold off

%% testirajuci podaci
X0t = data_test(data_test(:,end) == 0, 1:end-1);
X1t = data_test(data_test(:,end) == 1, 1:end-1);
Y0t = A'*X0t';
Y1t = A'*X1t';
%% kvadratni klasifikator za 2 dimenzije

U = zeros(6, length(data_trening));
for i = 1:length(Y0)
    U(:,i)=[-Y0(1,i)^2 -2*Y0(1,i)*Y0(2,i) -Y0(2,i)^2 -Y0(1,i) -Y0(2,i) -1]';
end

for i = 1:length(Y1)
    U(:,i+length(Y0))=[Y1(1,i)^2 2*Y1(1,i)*Y1(2,i) Y1(2,i)^2 Y1(1,i) Y1(2,i) 1]';
end
G = ones(length(data_trening),1);
%G(1:length(Y0)-1) = 0.7;
%G(length(Y0):end) = 1;
W = (U*U')^-1 *U*G;

q11=W(1);
q12 = W(2);
q22 = W(3);
v1 = W(4);
v2 = W(5);
v0 = W(6);

f = @(x1,x2) x1.^2*q11 + x2.^2*q22 + 2*q12*x1.*x2 + v1*x1 + v2*x2 + v0;

figure(2)
    plot(Y0(1,:),Y0(2,:),'o')
    hold on
    plot(Y1(1,:),Y1(2,:),'c*')
    %fimplicit(f,[xlim ylim])
    ezplot(f,[xlim ylim])
    title('Kvadratni klasifikator na bazi željenog izlaza')
    legend('Zdravi','Bolesni')
    hold off

%% testiranje
figure(3)
    plot(Y0t(1,:),Y0t(2,:),'o')
    hold on
    plot(Y1t(1,:),Y1t(2,:),'c*')
    %fimplicit(f,[xlim ylim])
    ezplot(f,[xlim ylim])
    title('Kvadratni klasifikator na bazi željenog izlaza - test')
    legend('Zdravi','Bolesni')
    hold off
    
%% Konfuziona matrica
% Testirajuci skup
[~, dim1] = size(Y0t);
[~, dim2] = size(Y1t);

odluke = [f(Y0t(1,:),Y0t(2,:))>0, f(Y1t(1,:),Y1t(2,:))>0]*1.0;
tacno = [zeros(1,dim1) ones(1,dim2)];
figure()
    plotconfusion(tacno,odluke);
    title('Konfuziona matrica - kvadratni klasifikator');
    xlabel('Tacna klasa'); ylabel('Estimirana klasa');
%% linearni klasifikator za 2 dimenzije

U = zeros(3, length(data_trening));
for i = 1:length(Y0)
    U(:,i)=[-Y0(1,i) -Y0(2,i) -1]';
end

for i = 1:length(Y1)
    U(:,i+length(Y0))=[Y1(1,i) Y1(2,i) 1]';
end
G = ones(length(data_trening),1);
%G(1:length(Y0)-1) = 0.7;
%G(length(Y0):end) = 1;

W = (U*U')^-1 *U*G;

v1 = W(1);
v2 = W(2);
v0 = W(3);

f = @(x1,x2) v1*x1 + v2*x2 + v0;

figure(4)
    plot(Y0(1,:),Y0(2,:),'o')
    hold on
    plot(Y1(1,:),Y1(2,:),'c*')
    %fimplicit(f,[xlim ylim])
    ezplot(f,[xlim ylim])
    title('Linearni klasifikator na bazi željenog izlaza')
    legend('Zdravi','Bolesni')
    hold off
    
%% testiranje
figure(5)
    plot(Y0t(1,:),Y0t(2,:),'o')
    hold on
    plot(Y1t(1,:),Y1t(2,:),'c*')
    %fimplicit(f,[xlim ylim])
    ezplot(f,[xlim ylim])
    title('Kvadratni klasifikator na bazi željenog izlaza - test')
    legend('Zdravi','Bolesni')
    hold off
    
%% Konfuziona matrica
% Testirajuci skup
[~, dim1] = size(Y0t);
[~, dim2] = size(Y1t);

odluke = [f(Y0t(1,:),Y0t(2,:))>0, f(Y1t(1,:),Y1t(2,:))>0]*1.0;
tacno = [zeros(1,dim1) ones(1,dim2)];
figure()
    plotconfusion(tacno,odluke);
    title('Konfuziona matrica - linearni klasifikator');
    xlabel('Tacna klasa'); ylabel('Estimirana klasa');
%% KNN klasifikator

X_test = data_test(:,1:end-1);


Y_test = A'*X_test';

distance = 0.5*10^-4;

num_healthy = 0;
num_infected = 0;

for y = Y_test
    healthy = 0;
    for y1 = Y0
        if (y1(1)-y(1))^2+(y1(2)-y(2))^2<distance^2
            healthy=healthy+1;
        end
    end
    infected = 0;
    for y1 = Y1
        if (y1(1)-y(1))^2+(y1(2)-y(2))^2<distance^2
            infected=infected+1;
        end
    end
    if healthy<=infected
        num_infected = num_infected+1;
        Y1_test(:,num_infected) = y;
    end
    if healthy>infected
        num_healthy = num_healthy+1;
        Y0_test(:,num_healthy) = y;
    end
end

figure(6)
    plot(Y0(1,:),Y0(2,:),'o')
    hold on
    plot(Y1(1,:),Y1(2,:),'c*')
    plot(Y0_test(1,:),Y0_test(2,:),'ro')
    plot(Y1_test(1,:),Y1_test(2,:),'g*')
    title('KNN :(')
