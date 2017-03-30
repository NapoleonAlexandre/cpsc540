function [w,f] = findMin(funObj,w,maxEvals,method,verbose,varargin)
% Find local minimizer of differentiable function

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g,H] = funObj(w, method, optTol, varargin{:});
funEvals = 1;

% choose method for backtracking
switch method
    case {'hermite', 'newton', 'hfn'}
        nextAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) hermiteNext(alpha, dirDeriv, f, f_new);
        resetAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) hermiteReset();
    case 'boydvandenberghe'
        nextAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) boyvanNext(alpha);
        resetAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) boyvanReset();
    case 'barzilaiborwein'
        nextAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) hermiteNext(alpha, dirDeriv, f, f_new);
        resetAlpha = @(alpha, dirDeriv, f, f_new, g, g_new, X, y, lambda) barborReset(alpha, g, g_new);
    case 'lipschitz'
        X = varargin{1};
        lambda = varargin{3};
        resetAlpha = @lipschitzReset;
end
alpha = 1;
if strcmp(method, 'lipschitz')
    alpha = resetAlpha(X, lambda);
end
backtracks = 0;
while 1
    %% Compute search direction
    if strcmp(method, 'newton') || strcmp(method, 'hfn')
        d = H;
    else
        d = g;
    end
    
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*d;
	[f_new,g_new,H_new] = funObj(w_new, method, optTol,varargin{:});
	funEvals = funEvals+1;
    
    dirDeriv = g'*d;
    while f_new > f - gamma*alpha*dirDeriv
        if strcmp(method,'lipschitz')
            break
        end
        if verbose
            fprintf('Backtracking...\n');
        end
        % Cubic Hermite
        % alpha = alpha^2*dirDeriv/(2*(f_new - f + alpha*dirDeriv));
        alpha = nextAlpha(alpha, dirDeriv, f, f_new, g, g_new);
        w_new = w - alpha*d;
        [f_new,g_new,H_new] = funObj(w_new, method, optTol,varargin{:});
        funEvals = funEvals+1;
        backtracks = backtracks + 1;
    end
    alphaFinal = alpha;

    %% Update step-size for next iteration
    if ~strcmp(method, 'lipschitz')
        alpha = resetAlpha(alpha, dirDeriv, f, f_new, g, g_new);
    end
    
    %% Sanity check on step-size
    if ~isLegal(alpha) || alpha < 1e-10 || alpha > 1e10
       alpha = 1; 
    end
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
    H = H_new;
	
    %% Test termination conditions
	optCond = norm(g,'inf');
    if verbose
        fprintf('%6d %6d %15.5e %15.5e %15.5e\n',funEvals, backtracks,alphaFinal,f,optCond);
    end
	
	if optCond < optTol
        if verbose
            fprintf('Problem solved up to optimality tolerance\n');
        end
		break;
	end
	
	if funEvals >= maxEvals
        if verbose
            fprintf('At maximum number of function evaluations\n');
        end
		break;
	end
end
end


% Cubic Hermite
function [nextAlpha] = hermiteNext(alpha, dirDeriv, f, f_new)
nextAlpha = alpha^2*dirDeriv/(2*(f_new - f + alpha*dirDeriv));
end
function [newAlpha] = hermiteReset()
newAlpha = 1;
end
% Boyd-Vandenberghe
function [nextAlpha] = boyvanNext(alpha)
nextAlpha = alpha/2;
end
function [newAlpha] = boyvanReset()
newAlpha = 1;
end
% Barzilai-Borwein
function [newAlpha] = barborReset(alpha, g, g_new)
v = g_new - g;
newAlpha = -alpha * (v' * g_new) / (v' * v);
end
% Lipschitz
function [newAlpha] = lipschitzReset(X, lambda)
L = .25 * max(eig(X' * X)) + lambda;
newAlpha = 1/L;
end


function [legal] = isLegal(v)
legal = sum(any(imag(v(:))))==0 & sum(isnan(v(:)))==0 & sum(isinf(v(:)))==0;
end