
%% ANIMAL TYPE (hf/normal) IS FIXED EFFECT, CHANNEL AND ANIMAL ID IS RANDOM EFFECT
T = readtable('entropy_all_numerical2.csv');

lme=fitlme(T, 'entropy_mean ~ animal_type + (1|channel) + (1|animal)')
lme=fitlme(T, 'entropy_std ~ animal_type + (1|channel) + (1|animal)')


