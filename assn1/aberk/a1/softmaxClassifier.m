function [model] = softmaxClassifier(X,y)

% Compute sizes
[n,d] = size(X);
k = max(y);
W = zeros(d,k); % Each column is a classifier
W(:) = findMin(@softmaxLoss, W(:), 500, 1, X, y, k);
model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [nll, g] = softmaxLoss(w, X, y, k)
% SOFTMAXLOSS computes the negative loglikelihood nll and the gradient g of the softmax loss function
[n, d] = size(X);
W = reshape(w, [d, k]);
XW = X*W;
Z = sum(exp(XW), 2);
ind=sub2ind ([n k],[1:n]',y);
nll=-sum(XW(ind)-log(Z));
g = zeros(d,k);
for c = 1:k
    g(:, c) = X' * (exp(XW(:,c))./Z - (y==c));
end
g = reshape(g, [d*k, 1]);
end