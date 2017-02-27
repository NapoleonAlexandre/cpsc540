function [ model ] = MixtureBernoulli(X,alpha,k,its)

[n,d] = size(X);
theta = mean(X);
model.predict = @predict;
model.sample = @sample;

pic = ones(k,1)/k;
mu = rand(k,d);

for j=1:its
    
    % E
    L = zeros(n,k);
    r = zeros(n,k);
    for i=1:k
        L(:,i) = sum(log(pic(i)*(mu(i,:).^X).*((1-mu(i,:)).^(1-X))),2);
    end
    
    Lmax = max(max(L));
    r = L-Lmax-log(sum(exp(L-Lmax),2));
    r = exp(r);
    
    %M step
    pic = (sum(r)+alpha-1)/(n+k*(alpha-1));
    for i=1:k
        mu(i,:) = (sum(spdiags(r(:,i),0,n,n)*X))/(n*pic(i));
    end
    
    model.predict = @predict;
    model.sample = @sample;
    model.mu = mu;
    model.pic = pic;
    model.r = r;

end
end


function nlls = predict(model, Xhat)
[t,d] = size(Xhat);
theta = model.theta;

nlls = -sum(prod0(Xhat,repmat(log(theta),[t 1])) + prod0(1-Xhat,repmat(log(1-theta),[t 1])),2);
end

function samples = sample(model,t)

d = length(theta);

samples = zeros(t,d);
for i = 1:t
    samples(i,:) = rand(1,d) < theta;
end
end
