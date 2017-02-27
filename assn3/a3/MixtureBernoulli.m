function [ model ] = MixtureBernoulli(X,alpha,k,its)

[n,d] = size(X);
%theta = mean(X);
theta = (sum(X,1)+alpha-1)/(n+alpha+beta-2);
model.theta = theta;
model.predict = @predict;
model.sample = @sample;

pic = ones(k,1)/k;
mu = ones(k,d)/(k*d);

for j=1:its


end

