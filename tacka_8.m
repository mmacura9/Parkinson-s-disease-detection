%% Marko
close all
clear
clc
f = fopen('parkinsons.data');
t = textscan(f, '%s');
fclose(f);

matrix = split(t{1},',');
var = str2double(matrix(2:end,2:end));
data = var;
X = data(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]);
y = data(:, 17);
%% Kristina
close all
clear
clc
matrix = table2array(readtable('parkinsonsData.csv'));
data = matrix;
X = data(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]);
y = data(:, 17);

%% Normalizacija
[M N] = size(data);
for i = 1:N
    data(:,i)= data(:,i)/max(data(:,i));
end

%% Podela obicna - deli 70/30 sve
% pokreces ili ovo ili sledecu
[N,~] = size(matrix);
ind = randperm(N);

Xtrain = X(ind(1:round(0.7*N)),:)'; 
Ytrain = y(ind(1:round(0.7*N)))';
Xtest = X(ind(round(0.7*N)+1:end),:)';
Ytest = y(ind(round(0.7*N)+1:end))';

%% Podela - posebno gleda bolesne i zdrave da podeli 70/30 oba

data1 =[X, y(:,1)];

X1 = data1(data1(:,end) == 0, 1:end-1);
X2 = data1(data1(:,end) == 1, 1:end-1);

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

%% Balansiranje - mozes da preskocis

diff_ob = size(X2_ob,1) - size(X1_ob,1);
diff_ob = round(0.6*diff_ob);
diff_t = size(X2_t,1) - size(X1_t,1);
diff_t = round(0.6*diff_t);

M_ob = mean(X1_ob, 1);
S_ob = cov(X1_ob, 1);
new_ob = mvnrnd(M_ob, S_ob, diff_ob);

X1_ob = [X1_ob; new_ob];

M_t = mean(X1_t, 1);
S_t = cov(X1_t, 1);
new_t = mvnrnd(M_t, S_t, diff_t);

X1_t = [X1_t; new_t];

%% spajanje - obavezno ako si delio posebno, inace NE

[dim1,~] = size(X1_ob);
[dim2,~] = size(X2_ob);
Xtrain = [X1_ob; X2_ob]';
Ytrain = [zeros(1,dim1) ones(1,dim2)];

[dim1,~] = size(X1_t);
[dim2,~] = size(X2_t);
Xtest = [X1_t; X2_t]';
Ytest = [zeros(1,dim1) ones(1,dim2)];
%% Jednoslojna mreza

nodes = [1 3 5 7 10 15 20 25]; % opcije za broj cvorova u skirvenom sloju

errors = zeros(1,length(nodes));
times = zeros(1,length(nodes));

for i=1:length(nodes)
    
    nodes_num = nodes(i);
    
    net = newff(Xtrain, Ytrain, nodes_num, {'logsig','satlins'});
    net.divideFcn = ''; % ne zelimo da on deli skup
    net.performFcn = 'mse'; 
    net.trainParam.goal = 1e-4;
    net.trainParam.epochs = 1000;
    net.trainParam.show = 10;
    
    [net, tr] = train(net, Xtrain, Ytrain);
    Yresult_train = round(sim(net, Xtrain));
    Yresult_test = round(sim(net, Xtest));
    
    figure()
        plotconfusion(Ytrain, Yresult_train);
        title(['Jednoslojna mreza sa ' num2str(nodes_num) ' cvorova - obucavajuci skup']);
    figure()
        plotconfusion(Ytest, Yresult_test);
        title(['Jednoslojna mreza sa ' num2str(nodes_num) ' cvorova - testirajuci skup']);
        
    C = confusionmat(Ytest, Yresult_test);
    errors(i) = (C(1,2)+C(2,1)) / length(Yresult_test);
    times(i) = tr.time(1,end);
end
%%
plot(nodes, 1-errors, '-*');
title('Ta?nost u zavisnosti od broja ?vorova - testiraju?i');
xlabel('Broj ?vorova u skirvenom sloju')
ylabel('Ta?nost')
%% Viseslojna mreza

nodes = [10 5 2]; % broj cvorova po sloju (kolko brojeva tolko slojeva + 1)

errors = zeros(1,length(nodes));
times = zeros(1,length(nodes));

net = newff(Xtrain, Ytrain, nodes, {'tansig','tansig','tansig','satlins'});
net.divideFcn = ''; % ne zelimo da on deli skup
net3.performParam.regularization = 0.05;
net.performFcn = 'mse'; 
net.trainParam.goal = 1e-4;
net.trainParam.epochs = 1000;
net.trainParam.show = 10;

[net, tr] = train(net, Xtrain, Ytrain);
Yresult_train = round(sim(net, Xtrain));
Yresult_test = round(sim(net, Xtest));

figure()
    plotconfusion(Ytrain, Yresult_train);
    title(['Troslojna mreza sa - obucavajuci skup']);
figure()
    plotconfusion(Ytest, Yresult_test);
    title(['Troslojna mreza sa - testirajuci skup']);

C = confusionmat(Ytest, Yresult_test);
errors(i) = (C(1,2)+C(2,1)) / length(Yresult_test);
times(i) = tr.time(1,end);

%% Rano zaustavljanje

nodes = [15 10 5]; % broj cvorova po sloju (kolko brojeva tolko slojeva + 1)

errors = zeros(1,length(nodes));
times = zeros(1,length(nodes));

net = newff(Xtrain, Ytrain, nodes, {'tansig','tansig', 'tansig', 'satlins'});
net.divideParam.trainRatio = 0.7; % podela u odnosu 70/20/10
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;  
net.performFcn = 'mse'; 
net.trainParam.epochs = 1000;
net.trainParam.show = 10;

[net, tr] = train(net, Xtrain, Ytrain);
Yresult_train = round(sim(net, Xtrain));
Yresult_test = round(sim(net, Xtest));

figure()
    plotconfusion(Ytrain, Yresult_train);
    title(['Rano zaustavljanje - obucavajuci skup']);
figure()
    plotconfusion(Ytest, Yresult_test);
    title(['Rano zaustavljanje - testirajuci skup']);

C = confusionmat(Ytest, Yresult_test);
errors(i) = (C(1,2)+C(2,1)) / length(Yresult_test);
times(i) = tr.time(1,end);

%% Balansirana tacnost
balanced_acc = (sum(Yresult_test == 1 & Ytest == 1)/sum(Ytest==1) +...
            sum(Yresult_test == 0 & Ytest == 0)/sum(Ytest==0))/2
            