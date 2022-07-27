function [conf_mat, class_loss] = entropy_energy_svm(data, label, repeat_time)
% In this function, we extract Shannon, Renyi, log_engery, approximate,
% sample and fuzzy entropy and exponential energy of x, x' and x'' as the
% input to SVM.
% In SVM, we use linear kernel.


% Apply 6th-order butterworth low-pass filter
fc = 60;
fs = 150;
[b,a] = butter(6,fc/(fs/2));
data = filter(b,a,data');
data = data';
% data = normalize(data',2);

data_p = diff(data,1,2);
% data_p = normalize(data_p,2);
data_pp = diff(data,2,2);
% data_pp = normalize(data_pp,2);

data_ori_feature = get_features(data);
data_p_feature = get_features(data_p);
data_pp_feature = get_features(data_pp);

% data_feature = [data_feature,data_p_feature,data_pp_feature];

% sum(sum(isinf(data_feature)))

% Apply SVM
% repeat_time = 8;
classLoss = zeros(repeat_time,127);
conf_mat = zeros(repeat_time,127,5,5); 

%%
% for j = 1:repeat_time
%     %     j
%     parfor i = 1:127
%         data_feature = [data_ori_feature(:,find(de2bi(i))),data_p_feature(:,find(de2bi(i))),data_pp_feature(:,find(de2bi(i)))];
%         %     SVMModel = fitcsvm(data_feature,label,'Standardize',true,'KernelFunction','linear');
%         t = templateSVM('Standardize',true,'KernelFunction','linear');
%         SVMModel = fitcecoc(data_feature,label,'Learners',t);
%         CVSVMModel = crossval(SVMModel,'KFold',5);
%         classLoss(j,i) = kfoldLoss(CVSVMModel);
%     end
% end

%%
options = statset('UseParallel',true);
for j = 1:repeat_time
%         j
    parfor i = 1:127
        data_feature = [data_ori_feature(:,find(de2bi(i))),data_p_feature(:,find(de2bi(i))),data_pp_feature(:,find(de2bi(i)))];
        %     SVMModel = fitcsvm(data_feature,label,'Standardize',true,'KernelFunction','linear');
        t = templateSVM('Standardize',true,'KernelFunction','linear');
        SVMModel = fitcecoc(data_feature,label,'Learners',t);
        CVSVMModel = crossval(SVMModel,'KFold',5);
        oofLabel = kfoldPredict(CVSVMModel,'Options',options);
        conf_mat(j,i,:,:) = confusionmat(label,oofLabel);
        classLoss(j,i) = kfoldLoss(CVSVMModel);
    end
end

class_loss = classLoss;

end

function data_feature = get_features(data)
data_len = length(data);
data_sd = std(data,[],2);
% Calculate Shannon Entropy, Approximate Entropy, Sample Entropy and Fuzzy
% Entropy
SE = zeros(data_len,1);
AE = zeros(data_len,1);
SamE = zeros(data_len,1);
FuzE = zeros(data_len,1);
for i = 1:data_len
    SE(i) = pentropy(data(i,:), linspace(0,2,size(data,2)), 'Instantaneous', false);
    AE(i) = approximateEntropy(data(i,:));
    
    SamE(i) = sampen(data(i,:), 2, 1, 0.2*data_sd(i));
    FuzE(i) = fuzzyen(data(i,:), 3, 3, 0.15*data_sd(i));
end

%Get power distribution
data_fft = fft(data, [], 2);
data_power = abs(data_fft(:,1:round(size(data,2)/2))).^2;
data_power = data_power./repmat(sum(data_power, 2),1,size(data_power,2));

% Calculate Renyi Entropy
alpha = 0.5;
RE = 1/(1-alpha)*log2(sum(data_power.^alpha,2));

% Calculate Log-Energy Entropy
LEE = sum((log2(data_power)).^2,2);

% Exponential Energy
EE = sum(exp(-data.^2),2);

% Combine feature together
data_feature = [RE,LEE,SE,AE,SamE,FuzE,EE];

end

