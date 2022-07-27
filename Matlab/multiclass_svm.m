clear
clc

# please change the "datapath" to the path where the data is
Files_1 = dir('datapath/data_1*.mat');
Files_2 = dir('datapath/data_2*.mat');
Files_3 = dir('datapath/data_3*.mat');
Files_4 = dir('datapath/data_4*.mat');
Files_5 = dir('datapath/data_5*.mat');

label = zeros(10000,1);
label(2001:4000) = 1;
label(4001:6000) = 2;
label(6001:8000) = 3;
label(8001:10000) = 4;
repeat_time = 10;
class_loss = zeros(81,repeat_time,127);
confmat_total = zeros(81,repeat_time,127,5,5);

for i = 1:81
    i
    load(Files_1(i).name)
    load(Files_2(i).name)
    load(Files_3(i).name)
    load(Files_4(i).name)
    load(Files_5(i).name)
    data = [data_1;data_2;data_3;data_4;data_5];
%     [confmat_total, class_loss] = entropy_energy_svm(data, label,repeat_time);
    [confmat_total(i,:,:,:,:), class_loss(i,:,:)] = entropy_energy_svm(data, label,repeat_time);
end

save('confmat_total.mat','confmat_total')
save('class_loss.mat','class_loss')

% Mdl = fitcecoc(X,Y);
% Mdl = crossval(Mdl,'KFold',5);
% classLoss = kfoldLoss(Mdl)
