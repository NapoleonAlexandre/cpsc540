function model = generativeGaussianSSL(Xtrain,Ytrain,Xunlabeled,k,its,hard)

[n,d] = size(Xtrain);
[t,~] = size(Xunlabeled);
model = generativeGaussian(Xtrain,Ytrain,k);
Q = inf;
dQ = inf;
tol = 1e-2;
ct = 0;
model.theta0 = model.theta;
r = zeros(t,k);

%repeat the algorithm its times
for j=1:its
    
    % E step
    %For each class i update r
    for i=1:k
        r(:,i) = mvnpdf(Xunlabeled,model.mu{i}',model.SIGMA{i})*model.theta(i);
    end
    %Normalization method for sum = 1, dependent of hard of soft-EM
    if hard
        r = double(bsxfun(@eq, r, max(r, [], 2)));
    else
        r = r./sum(r,2);
    end
    
    %M step
    %For each class i
    for i=1:k
        ind = find(Ytrain == i);
        
        %Compute average mu_i
        model.mu{i} = (1/(n*model.theta0(i)+sum(r(:,i))))*(sum(Xtrain(ind,:))' ...
            + sum(spdiags(r(:,i),0,t,t)*Xunlabeled)');
        
        %Compute cavariance matrix SIGMA_i
         model.SIGMA{i} = (1/(n*model.theta0(i)+sum(r(:,i))))*...
            (((Xtrain(ind,:)'-model.mu{i})*(Xtrain(ind,:)'-model.mu{i})')+ ...
            ((Xunlabeled'-model.mu{i})*spdiags(r(:,i),0,t,t)*(Xunlabeled'-model.mu{i})'));

        %Compute theta
        model.theta(i) = (n*model.theta0(i)+sum(r(:,i)))/(n+t);
        
    end

end
model.r = r;
end

