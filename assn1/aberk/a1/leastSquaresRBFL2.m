function [model] = leastSquaresRBFL2(X,y,lambda,sigma)
% Compute sizes
[n,d] = size(X);
% Add bias variable
Z = rbfBasis(X, [], sigma);
% Solve least squares problem
w = (Z'*Z + lambda*eye(n))\Z'*y;

model.X = X;
model.w = w;
model.lambda = lambda;
model.sigma = sigma;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Zhat = rbfBasis(Xhat, model.X, model.sigma);
yhat = Zhat*model.w;
end

function [Z] = rbfBasis(X1,X2,sigma)
% X1 = query points
% X2 = training points, or X2 = X1
if isempty(X2)
    X2 = X1;
end
if isempty(sigma)
    sigma = 1;
end
n1 = size(X1,1);
n2 = size(X2,1);
d = size(X1,2);
den = 1/sqrt(2*pi*sigma^2);
D = X1.^2*ones(d,n2) + ones(n1,d)*(X2').^2 - 2*X1*X2';
Z = den*exp(-D/(2*sigma^2));
end