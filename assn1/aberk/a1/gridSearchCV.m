function [bestModel, score] = gridSearchCV(func, X, y, parms, folds)
% GRIDSEARCHCV performs nfold cross-validation of func on a paramater list parms
% passed as a cell-array. Data and labels are given as X and y, resp.
parms = allcomb(parms{:});
[plLen, ~] = size(parms); % parms list length, num parameters (e.g. (lambda, sigma))
modelList = cell(plLen,1);
errorList = zeros(1, plLen);
for j = 1:plLen
    display(sprintf('Iteration %d of %d.', j, plLen));
    parm = num2cell(parms(j,:));
    [modelList{j}, errorList(j)] = CV(func,X,y,parm,folds);
end
score = min(errorList);
bestModel = modelList{errorList == score};
end


function [model, score] = CV(func, X, y, parms, folds)
% CV performs nfolds-nfold cross validation
% Note: parms is a cell array
%       folds must be denoted using 1, 2, ..., nfolds

% n observations; d-dimensional feature space
[n, d] = size(X);
% default: 5-fold cross-validation
if isempty(folds) || (numel(folds)==1 && folds < 2)
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

squaredTestError = zeros(1, nfolds);

for j = 1:nfolds
    model = func(X(folds~=j,:), y(folds~=j), parms{:});
    yhat = model.predict(model, X(folds==j,:));
    squaredTestError(j) = sum((yhat-y(folds==j)).^2)/length(yhat);
end
score = mean(squaredTestError);
end

function folds = getFolds(n, nfolds)
% GETFOLDS returns a vector giving fold identities for each element.
folds = zeros(1, n);
perm = randperm(n);
for j = 1:nfolds
    j0 = round(1 + (j-1)*n/nfolds);
    j1 = round(j*n/nfolds);
    folds(perm(j0:j1)) = j;
end
end
