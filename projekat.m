close all
clear
clc
f = fopen('parkinsons.data');
t = textscan(f, '%s');
fclose(f);

matrix = split(t{1},',');
val = str2double(matrix(2:end,2:end));
data = val(:,[1,3,5,9,13,18,19,20,21,22, 17]);
cor = corrcoef(data);

%% Correlation feature selection
k=10;
rzi = mean(cor(end,1:end-1));
rii = (sum(sum(cor(1:end-1,1:end-1)))-10)/10/9; %PROVERI
r = k*rzi/sqrt(k+k*(k-1)*rii);

%% Information Gain
[M N] = size(val);
p1 = sum(data(1:end,end))/M;
p0 = 1 - p1;

Info_D = - p0*log2(p0) - p1*log2(p1)
