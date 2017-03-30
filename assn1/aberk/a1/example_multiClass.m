clear all
close all
load multiData.mat

% Data is already roughly standardized, but let's add bias
[n,d] = size(X);
X = [ones(n,1) X];

% Fit least-squares classifier
model = logLinearClassifier(X,y);
model2 = softmaxClassifier(X,y);

% Compute validation error
t = size(Xvalidate,1);
Xvalidate = [ones(t,1) Xvalidate];
yhat = model.predict(model,Xvalidate);
yhat2 = model2.predict(model2, Xvalidate);
errors = sum(yvalidate ~= yhat)/t
errors2 = sum(yvalidate ~= yhat2)/t

% Plot result
k = max(y);
multiClassifier2Dplot(X,y,k,model);
title('log-linear classifier');
multiClassifier2Dplot(X,y,k,model2);
title('softmax classifier');