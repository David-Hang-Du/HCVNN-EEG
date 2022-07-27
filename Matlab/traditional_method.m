clear
clc

load noise_fix.mat
noise_fix = 1-class_loss;
noise_fix_mean = mean(noise_fix);
[m,I] = max(noise_fix_mean);
m
noise_fix_std = std(noise_fix);
noise_fix_std(I)


load noise_moving.mat
noise_moving = 1-class_loss;
noise_moving_mean = mean(noise_moving);
[m,I] = max(noise_moving_mean);
m
noise_moving_std = std(noise_moving);
noise_moving_std(I)

load phase_fix.mat
phase_fix = 1-class_loss;
phase_fix_mean = mean(phase_fix);
[m,I] = max(phase_fix_mean);
m
phase_fix_std = std(phase_fix);
phase_fix_std(I)


load phase_moving.mat
phase_moving = 1-class_loss;
phase_moving_mean = mean(phase_moving);
[m,I] = max(phase_moving_mean);
m
phase_moving_std = std(phase_moving);
phase_moving_std(I)




