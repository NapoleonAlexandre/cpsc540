function [model] = logisticL2(X,y,lambda)
%

% method \in {'original', 'coordinate1', 'coordinate2', 'coordinate3'}
%  ... where ...
% 'original' : no modifications to the code
% 'coordinate1' : runtime is now O(n*d*Lc/mu*log(1/epsilon))
% 'coordinate2' : Lipschitz sampling of the index j; 
%                 uses the jth Lipschitz constant.
% 'coordinate3' : uniform sampling of the index j; 
%                 uses the jth Lipschitz constant.
method = 'coordinate3';

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];
d = d+1;

% Initial values of regression parameters
w = zeros(d,1);

% Optimizaion parameters
maxPasses = 500;
progTol = 1e-4;
if strcmp(method, 'coordinate1')
    L = .25*max(sum(X.^2,1)) + lambda;
elseif strcmp(method, 'coordinate2') || strcmp(method, 'coordinate3')
    Lvec = .25 * sum(X.^2, 1) + lambda;
    p_Lvec = Lvec;
    p_Lvec = p_Lvec./sum(p_Lvec);
else
    L = .25*max(eig(X'*X)) + lambda;
end

if strcmp(method(1:end-1), 'coordinate')
    Xw = X*w;
    Xw_old = Xw;
    doCoord = 1;
else
    doCoord = 0;
end
w_old = w;

for t = 1:maxPasses*d
    % Choose variable to update 'j'
    if strcmp(method, 'coordinate2')
        j = sampleDiscrete(p_Lvec);
        L = Lvec(j);
    elseif strcmp(method, 'coordinate3')
        j = randi(d);
        L = Lvec(j);
    else
        j = randi(d);
    end
    
    % Compute partial derivative 'g_j'
    if doCoord
        yXw = y.*Xw;
    else
        Xw = X*w;
        yXw = y.*Xw;
    end
    sigmoid = 1./(1+exp(-yXw));
    if doCoord
        g_j = -X(:,j)'*(y.*(1-sigmoid)) + lambda*w(j);
    else
        g = -X'*(y.*(1-sigmoid)) + lambda*w;
        g_j = g(j);
    end
    
    % Update variable
    if doCoord
        Xw = Xw - (1/L)*g_j*X(:,j);
    end
        w(j) = w(j) - (1/L)*g_j;
    
    % Check for lack of progress after each "pass"
    if mod(t,d) == 0
        change = norm(w-w_old,inf);
        if doCoord
            fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/d,logisticL2_loss(w,Xw,y,lambda),change);
        else
            fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/d,logisticL2_loss(w,X,y,lambda),change);
        end
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        w_old = w;
    end
end

model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end


