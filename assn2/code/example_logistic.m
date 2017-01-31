clear model;
load logisticData.mat

tic
model = logisticL2(X,y,1);
toc


% original:
% Passes = 500, function = 1.4735e+02, change = 0.0003
% Elapsed time is 17.793978 seconds.
% 
% coordinate1:
% Passes = 500, function = 1.4723e+02, change = 0.0005
% Elapsed time is 3.021629 seconds.
% 
% coordinate2:
% Passes = 291, function = 1.4330e+02, change = 0.0001
% Parameters changed by less than progTol on pass
% Elapsed time is 2.132973 seconds.
% 
% coordinate3:
% Passes = 136, function = 1.4091e+02, change = 0.0001
% Parameters changed by less than progTol on pass
% Elapsed time is 0.823345 seconds.

