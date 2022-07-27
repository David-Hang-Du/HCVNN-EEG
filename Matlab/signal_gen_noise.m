clear
clc
% figure()
% a = zeros(1,300);
% for i = 1:100
% mysin = Makinen (300, 1, 250, 100);
% a = a + mysin;
% end
% plot(a)
% subplot(3,1,1)
% plot (mysin);
% subplot(3,1,2)
% plot(abs(fft(mysin)))
% subplot(3,1,3)
% plot(angle(fft(mysin)))

total_noise = zeros(5000,300);
total_noise_fft = zeros(5000,150);
total_signal = zeros(5000,300);
total_signal_fft = zeros(5000,150);

for i = 1:1
    % figure()
    mynoise = noise(300, 1, 150);
    mynoise_fft = fft(mynoise);
    total_noise(i,:) = mynoise;
    total_noise_fft(i,:) = mynoise_fft(2:151);
    
%     mypeak = peak (300, 1, 150, 5, randi([50,250]));
    mypeak = peak(300, 1, 150, 5, 150);
    mysignal = mypeak + mynoise;
    mysignal_fft = fft(mysignal);
    total_signal(i,:) = mysignal;
    total_signal_fft(i,:) = mysignal_fft(2:151);
end
% plot(total_signal(1,:))
% hold on
% fc = 60;
% fs = 150;
% [b,a] = butter(6,fc/(fs/2));
% total_noise = filter(b,a,total_noise');
% total_noise = total_noise';
% total_signal = filter(b,a,total_signal');
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


% % data_label = zeros(10000,1);
% % data_label(1:5000) = 1;


% classLoss = zeros(1,10);
% for i = 1:10
%     SVMModel = fitcsvm(data_total,data_label,'KernelFunction','linear');
%     CVSVMModel = crossval(SVMModel,'KFold',5);
%     classLoss(i) = kfoldLoss(CVSVMModel)
% end
% mean(classLoss)
% std(classLoss)


% % data = [total_noise;total_signal];
% % repeat_time = 10;
% % class_loss = entropy_energy_svm(data, data_label, repeat_time);


%%
figure()
set(gcf, 'Position',  [100, 100, 800, 200])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,2,300), mynoise)
xlabel('Time(ms)')
% title('Noise')

figure()
set(gcf, 'Position',  [100, 100, 800, 200])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,2,300), mypeak)
xlabel('Time(ms)')
% title('Fixed peak location')

figure()
set(gcf, 'Position',  [100, 100, 800, 200])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,2,300), mysignal)
xlabel('Time(ms)')
% title('Noise + Peak')
%%
mynoise = noise(300, 1, 150);
mynoise_fft = fft(mynoise);
total_noise(i,:) = mynoise;
total_noise_fft(i,:) = mynoise_fft(2:151);

mypeak1 = peak (300, 1, 150, 5, 70);
% mypeak = peak (300, 1, 150, 5, 150);
mysignal = mypeak1 + mynoise;
mypeak2 = peak (300, 1, 150, 5, 170)+ peak (300, 1, 150, 5, 220);
mysignal_fft = fft(mysignal);
total_signal(i,:) = mysignal;
total_signal_fft(i,:) = mysignal_fft(2:151);

figure()
set(gcf, 'Position',  [100, 100, 800, 300])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,2,300),mynoise,'color','k')
% title('Noise')
xlabel('Time(ms)')
% label_h = ylabel('random peak location','Rotation',0)
% label_h.Position(1) = -100; % change horizontal position of ylabel
% label_h.Position(2) = 0; % change vertical position of ylabel
% hYLabel = get(gca,'YLabel');
% set(hYLabel,'rotation',0,'VerticalAlignment','middle')

figure()
set(gcf, 'Position',  [100, 100, 800, 300])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,1,150),mypeak1(1:150),linspace(1,2,150),mypeak2(151:end),'--','color','k')
% title('Random peak location')
xlabel('Time(ms)')


figure()
set(gcf, 'Position',  [100, 100, 800, 300])
tiledlayout(1,1,'TileSpacing','compact','Padding', 'tight')
nexttile
plot(linspace(0,2,300),mysignal,'color','k')
% title('Noise + Peak')
xlabel('Time(ms)')


% save('total_noise_cp.mat', 'total_noise');
% save('total_signal_cp.mat', 'total_signal');
% save('total_noise_fft_cp.mat', 'total_noise_fft');
% save('total_signal_fft_cp.mat', 'total_signal_fft');
