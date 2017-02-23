function [nll,g,H] = logisticL2_loss(w,X,y,lambda)
if iscolumn(X)
    yXw = y.*X; % because X is actually Xw
else
    yXw = y.*(X*w);
end

% Function value
nll = sum(log(1+exp(-yXw))) + (lambda/2)*(w'*w);

% Gradient
sigmoid = 1./(1+exp(-yXw));
g = -X'*(y.*(1-sigmoid)) + lambda*w;
end