
% Clear variables and close figures
clear all
close all

% Load data
load nonLinear.mat % Loads {X,y,Xtest,ytest}
[n,d] = size(X);
[t,~] = size(Xtest);

% parameter settings
lambda = logspace(-7, -5, 30);
sigma = logspace(-1,0,30);

% Train least squares model on training data
[model, score] = gridSearchCV(@leastSquaresRBFL2, X, y, {lambda, sigma}, 10);
%%
display(sprintf('best squared test error %f for (lambda, sigma) = (%e, %f)', score, model.lambda, model.sigma));

%% Plot model
figure(1);
plot(X,y,'b.');
hold on
plot(Xtest,ytest,'g.');
Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
yhat = model.predict(model,Xhat);
ypredict = model.predict(model, Xtest);
display(sprintf('Test MSE: %15.5g',mean((ypredict - ytest).^2)));
plot(Xhat,yhat,'r');
ylim([-300 400]);
set(gcf, 'Color', [1,1,1]);
set(gca, 'FontSize', 16);
xlabel('Xhat');
ylabel('yhat');
title(sprintf('(lambda, sigma) = (%15.5e, %15.5e)\nMSE Test = %15.5g', ...
    model.lambda, model.sigma, mean((ypredict - ytest).^2)));
hold off