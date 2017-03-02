function [ model ] = densityBernoulli(X,alpha,beta)

[n,d] = size(X);
%theta = mean(X);
theta = (sum(X,1)+alpha-1)/(n+alpha+beta-2);
model.theta = theta;
model.predict = @predict;
model.sample = @sample;
end

function nlls = predict(model, Xhat)
[t,d] = size(Xhat);
theta = model.theta;

nlls = -sum(prod0(Xhat,repmat(log(theta),[t 1])) + prod0(1-Xhat,repmat(log(1-theta),[t 1])),2);
end

function samples = sample(model,t)
theta = model.theta;
d = length(theta);

samples = zeros(t,d);
for i = 1:t
    samples(i,:) = rand(1,d) < theta;
end
end