% Housekeeping
clear all; close all; clc;
addpath('../a3/'); % make logdet discoverable
rng(2112) % set the seed

% Parameters
N = 10; % number of rows/cols
A = randn(N); % normal random matrix
s = rand(N,1); % singular values

% Build SPD matrix
M = zeros(N);
for i = 1:N
    M = M + s(i)*A(i, :)'*A(i,:);
end

% Computation
l = logdet(M, 'error');
scaledEigenProduct = prod(eig(M))/N;

% Result
display(sprintf(['exp(logdet(M))/N = %5.4g\n',...
                 '  prod(eig(M))/N = %5.4g'],...
                 exp(l)/N, scaledEigenProduct));