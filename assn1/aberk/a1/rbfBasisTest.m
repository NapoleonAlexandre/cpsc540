function rbfBasisTest(X1, X2, sigma)
Z_T = rbfBasis_T(X1, X2, sigma);
Z_A = rbfBasis_A(X1, X2, sigma);
squaredError = (Z_T(:) - Z_A(:)).^2;
displaystr = sprintf('squared error: %.3e;\nmean squared error: %.3e', sum(squaredError), mean(squaredError));
display(displaystr);

ff = figure;
suptitle(displaystr);
subplot(121);
imagesc(Z_T);
colorbar;
title('Matrix construction');
set(gca, 'FontSize', 16);
subplot(122);
imagesc(Z_A);
colorbar;
title('Element-wise construction');
set(gca, 'FontSize', 16);
set(gcf, 'Color', [1,1,1]);
input('Close plot? [Enter]');
close(ff);
end

function [Z] = rbfBasis_T(X1,X2,sigma)
if isempty(X2)
    X2 = X1;
end
if isempty(sigma)
    sigma = 1;
end
n1 = size(X1,1);
n2 = size(X2,1);
d = size(X1,2);
den = 1/sqrt(2*pi)/sigma;
D = X1.^2*ones(d,n2) + ones(n1,d)*(X2').^2 - 2*X1*X2';
Z = den*exp(-D/(2*sigma^2));
end

function [Z] = rbfBasis_A(Y, X, sigma)
% RBFBASIS computes the radial basis function matrix from matrices X and Y
% given by Z(j,k) = exp(-|xk-yj|^2/(2*sigma^2)).
% standard deviation is positive
if sigma <= 0
    error('sigma must be positive');
end
% Pass empty Y to construct symmetric rbf matrix
if isempty(X)
    % display('Y is empty; using X');
    X = Y;
end
% Default value for sigma is 1
if isempty(sigma)
    sigma = 1;
end
numX = size(X,1);
numY = size(Y,1);
Z = zeros(numY, numX);
for j = 1:numY
    for k = 1:numX
        Z(j,k) = phi(Y(j, :) - X(k, :), sigma);
    end
end
end

function p = phi(x, sigma)
% PHI is the Normal distribution's probability density function 
% with mean zero and standard deviation sigma.
p = exp(-norm(x, 2)./(2.*sigma.^2))/sqrt(2*pi)/sigma;
end