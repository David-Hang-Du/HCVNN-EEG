ce% clear
% clc
subplot(2,1,1)
plot(signal_noreset(1,:))
subplot(2,1,2)
plot(signal_reset(1,:))
% subplot(3,1,3)
% plot(mysignal)


% load total_noise.mat
% load total_noise_fft.mat
% load total_signal.mat
% load total_signal_fft.mat
% figure()
% subplot(2,1,1)
% plot(sum(total_noise)')
% subplot(2,1,2)
% plot(angle(total_signal_fft(1,:))')

load signal_reset_cp.mat
load signal_noreset.mat
load signal_noreset_fft.mat
load signal_reset_fft_cp.mat

for N = 1:10
    mynoise = noise(300, 1, 150);
    figure()
    subplot(2,1,1)
    plot(signal_reset(N,:)+4*mynoise)
    subplot(2,1,2)
    plot(abs(fft(signal_reset(N,:)+4*mynoise)))
end