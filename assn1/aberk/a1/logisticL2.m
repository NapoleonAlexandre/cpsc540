function [model] = regularizedLogisticRegression(X,y,lambda,method)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Initial values of regression parameters
w = zeros(d+1,1);

% Solve logistic regression problem
maxIter = 500;
verbose = 1;
w = findMin(@objective,w,maxIter,method,verbose,X,y,lambda);

model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function [nll,g,H] = objective(w,method,optTol,X,y,lambda)
yXw = y.*(X*w);

% Function value
nll = sum(log(1+exp(-yXw))) + (lambda/2)*(w'*w);

% Gradient
sigmoid = 1./(1+exp(-yXw));
g = -X'*(y.*(1-sigmoid)) + lambda*w;

% Hessian
if strcmp(method, 'newton')
    [n, d] = size(X);
    H = lambda*eye(d);
    for j = 1:n
        H = H + (X(j,:)'*X(j,:))./(2 + 2*cosh(yXw(j)));
    end
    H = H\g;
elseif strcmp(method, 'hfn')
    Hv = @(v) Hvfunc(w,v,X,y,lambda);
    H = pcg(Hv, g, optTol);
else
    H = [];
end
end

function [Hv] = Hvfunc(w,v,X,y,lambda)
yXw = y.*(X*w);
sigmoid = 1./(1+exp(-yXw));
Hv = X'*(diag(sparse(sigmoid.*(1-sigmoid)))*(X*v)) + lambda*v;
end

