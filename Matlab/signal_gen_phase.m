clear
clc
% mysin = phasereset (300, 1, 150, 5, 5, 115);
% mysin = Makinen (300, 1, 150, 200);
%
% figure()
% subplot(3,1,1)
% plot(mysin);
% subplot(3,1,2)
% plot(abs(fft(mysin)));
% subplot(3,1,3)
% plot(angle(fft(mysin)));

% signal = phasereset (300, 2, 150, 5, 5, 150, 0);
% figure()
% signal = Makinen(300, 1, 150, 4, 16, 150);
% signal_fft = fft(signal);
% signal_angle = angle(signal_fft);
% % mysin = Makinen (300, 1, 150, 150);
% subplot(3,1,1)
% plot(signal)
% subplot(3,1,2)
% plot(abs(signal_fft))
% subplot(3,1,3)
% plot(signal_angle)
%
% figure()
% signal = Makinen_noreset(300, 1, 150, 4, 16);
% signal_fft = fft(signal);
% signal_angle = angle(signal_fft);
% % mysin = Makinen (300, 1, 150, 150);
% subplot(3,1,1)
% plot(signal)
% subplot(3,1,2)
% plot(abs(signal_fft))
% subplot(3,1,3)
% plot(signal_angle)

% signal = No_phasereset (300, 1, 150, 4, 16);
% signal = Makinen (300, 1, 150, 100);
% plot(signal)

%  a = (1:256)/256*6*pi;
%  signal = zeros(1,256);
%  for i = 1:4
%      signal = signal + sin(i*a);
%  end
%  plot(signal)
signal_reset_noise = zeros(5000,300);
signal_reset_fft = zeros(5000,150);
signal_noreset_noise = zeros(5000,300);
signal_noreset_fft = zeros(5000,150);
rng(5)
for i = 1:5000
%     mynoise_1 = noise(300, 1, 150);
%     mynoise_2 = noise(300, 1, 150);
    mynoise_3 = noise(300, 1, 150);
%     res_pos = randi([50,250]);
    res_pos = 150;
    sig_reset_tmp = Makinen(300, 1, 150, 4, 16, res_pos);
%     sig_reset_tmp_fixed = Makinen(300, 1, 150, 4, 16, 150);
    signal_reset_noise(i,:) = sig_reset_tmp +2*mynoise_3;
%     sig_reset_fft = fft(sig_reset_tmp);
%     signal_reset(i,:) = sig_reset_tmp;
%     signal_reset_fft(i,:) = sig_reset_fft(2:151);
    
    
    
    %%    
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300), sig_reset_tmp_fixed)
%     xlabel('Time(ms)')
%     annotation('textarrow', [0.5 0.5], [0.25 0.35], 'String','Location of phase resetting')
% %     title('Signal with fixed location of phase resetting')
% 
%     %%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),mynoise_1)
%     xlabel('Time(ms)')
% %     title('Noise')
%     %%%%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),sig_reset_tmp_fixed+2*mynoise_1)
%     xlabel('Time(ms)')
%     title('Signal + noise')
    
    
    %%     nexttile
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),sig_reset_tmp)
%     xlabel('Time(ms)')
%     annotation('textarrow', [0.7805 0.7805], [0.4 0.48], 'String','Random location of phase reset')
% %     title('Signal with random location of phase resetting')
%     %%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),mynoise_2)
%     xlabel('Time(ms)')
% %     title('Noise')
%     %%%%%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),sig_reset_noise)
%     xlabel('Time(ms)')
%     title('Signal + noise')
%     
    %%
    sig_noreset_tmp = Makinen_noreset(300, 1, 150, 4, 16);
    signal_noreset_noise(i,:) = sig_noreset_tmp + 2*mynoise_3;
%     %     nexttile
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),sig_noreset_tmp)
%     xlabel('Time(ms)')
% %     title('Signal without phase reset')
%     %%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),mynoise_3)
%     xlabel('Time(ms)')
% %     title('Noise')
%     %%%%%%%%%%%%%%%%%%%%%
%     figure()
%     set(gcf, 'Position',  [100, 100, 800, 300])
%     tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
%     nexttile
%     plot(linspace(0,2,300),sig_noreset_noise)
%     xlabel('Time(ms)')
%     title('Signal + noise')
    %     sig_noreset_fft = fft(sig_noreset_tmp);
    %     signal_noreset(i,:) = sig_noreset_tmp;
    %     signal_noreset_fft(i,:) = sig_noreset_fft(2:151);
end

% fc = 60;
% fs = 150;
% [b,a] = butter(6,fc/(fs/2));
% total_noise = filter(b,a,signal_reset_noise');
% total_noise = total_noise';
% total_signal = filter(b,a,signal_noreset_noise');
% total_signal = total_signal';
% % plot(total_signal(1,:))
% 
% tn_eng = sum(exp(-total_noise.^2),2);
% ts_eng = sum(exp(-total_signal.^2),2);
% 
% tn_p_eng = sum(exp(-diff(total_noise,1,2).^2),2);
% ts_p_eng = sum(exp(-diff(total_signal,1,2).^2),2);
% 
% tn_pp_eng = sum(exp(-diff(total_noise,2,2).^2),2);
% ts_pp_eng = sum(exp(-diff(total_signal,2,2).^2),2);
% 
% tn = [tn_eng,tn_p_eng,tn_pp_eng];
% ts = [ts_eng,ts_p_eng,ts_pp_eng];
% data_total = [tn;ts];
data_label = zeros(10000,1);
data_label(1:5000) = 1;
% classLoss = zeros(1,10);
% for i = 1:10
%     SVMModel = fitcsvm(data_total,data_label,'KernelFunction','linear');
%     CVSVMModel = crossval(SVMModel,'KFold',5);
%     classLoss(i) = kfoldLoss(CVSVMModel)
% end
% mean(classLoss)
% std(classLoss)
data = [signal_reset_noise;signal_noreset_noise];
class_loss = entropy_energy_svm(data, data_label, 10);

save('phase_fix.mat','class_loss')


% save('signal_reset_noise_cp.mat', 'signal_reset')
% save('signal_reset_fft_noise_cp.mat', 'signal_reset_fft')
% save('signal_noreset_noise.mat', 'signal_noreset')
% save('signal_noreset_fft_noise.mat', 'signal_noreset_fft')