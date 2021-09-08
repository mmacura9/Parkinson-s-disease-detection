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
cor = corrcoef(data);

[M N] = size(data);

for i = 1:N
    data(:,i)= data(:,i)/max(data(:,i));
end


% %% Kristina
% close all
% clear
% clc
% matrix = table2array(readtable('parkinsonsData.csv'));
% data = matrix(:,[1,3,5,9,13,18,19,20,21,22,17]);
% cor = corrcoef(data);

%% Correlation feature selection
k=10;
rzi = mean(cor(end,1:end-1));
rii = (sum(sum(cor(1:end-1,1:end-1)))-10)/10/9; %PROVERI
r = k*rzi/sqrt(k+k*(k-1)*rii);

%% Information Gain
data = round(data,3,'significant');% podeli sa max

p1 = sum(data(1:end,end))/M;
p0 = 1 - p1;

Info_D = - p0*log2(p0) - p1*log2(p1);
Info_DA(1:10) = zeros(1,10); 


for i = 1:10
    K = data(:,i); 
    elemK = unique(K);
    total_0 = sum(data(:,end)==0);
    total_1 = sum(data(:,end)==1);
    for j = 1:length(elemK) 
        if sum(K==elemK(j) & data(:,end)==0)>0 
            absD = length(K);
            absDj = sum(K==elemK(j));
            p1 = sum(K==elemK(j) & data(1:end,end)==1)/absDj;
            p0 = 1 - p1;
            if p1==1 || p1 ==0
                InfoDj =0;
            else
                InfoDj = -p0*log2(p0)-p1*log2(p1);
            end
            Info_DA(i) = Info_DA(i) + absDj/absD*InfoDj;
        end
    end 
end 

IG = Info_D-Info_DA