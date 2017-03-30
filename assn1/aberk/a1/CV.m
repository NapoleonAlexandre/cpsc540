function [model, score] = CV(func, X, y, parms, folds)
% CV performs nfolds-nfold cross validation
% Note: parms is a cell array
%       folds must be denoted using 1, 2, ..., nfolds

% n observations; d-dimensional feature space
[n, d] = size(X);
% default: 5-fold cross-validation
if iempty(folds) || (numel(folds)==1 && folds < 2)
    nfolds = 5;
    folds = getFolds(n, nfolds);
elseif numel(folds) > 1
    if length(folds) ~= n
        error('fold vector must have length equal to number of observations');
    end
    nfolds = max(folds);
elseif numel(folds) == 1 && folds >= 2
    nfolds = folds;
    folds = getFolds(n, nfolds);
end

squaredTestError = zeros(1, folds);

for j = 1:nfolds
    model = func(X(folds~=j,:), y(folds~=j), parms{:});
    yhat = model.predict(model, X(folds==j,:));
    squaredTestError(j) = sum((ytest-yhat).^2)/length(ytest);
end
score = mean(squaredTestError);
end

function [bestModel, score] = gridSearchCV(func, X, y, parms, folds)
parms = allcomb(parms{:});
[plLen, ~] = size(parms); % parms list length, num parameters (e.g. (lambda, sigma))
modelList = cell(plLen,1);
errorList = zeros(1, plLen);
for j = 1:size(plLen)
    [modelList{j}, errorList(j)] = CV(func,X,y,num2cell(parms(j,:)),folds);
end
score = min(errorList);
bestModel = modelList{errorList == score};
end


function folds = getFolds(n, nfolds)
folds = zeros(1, n);
perm = randperm(n);
for j = 1:nfolds
    j0 = round(1 + (j-1)*n/nfolds);
    j1 = round(j*n/nfolds);
    folds(perm(j0:j1)) = j;
end
end
