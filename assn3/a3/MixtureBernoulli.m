function [ model ] = MixtureBernoulli(X,alpha,beta,k,its)

[n,d] = size(X);
theta = mean(X);
model.predict = @predict;
model.sample = @sample;

pic = ones(k,1)/k;
mu = 0.25+0.5*rand(k,d);

for j=1:its
    
    % E
    L = zeros(n,k);
    r = zeros(n,k);
    for i=1:k
        L(:,i) = sum(log(pic(i)*(mu(i,:).^X).*((1-mu(i,:)).^(1-X))),2);
    end
    
    Lmax = max(max(L));
    %LogSumExp trick
    r = L-Lmax-log(sum(exp(L-Lmax),2));
    r = exp(r);
    
    %M step
    z = sum(r);
    for i=1:k
        mu(i,:) = (sum(spdiags(r(:,i),0,n,n)*X)+beta-1)/(z(i)+k*(beta-1));
    end
    pic = (z+alpha-1)/(n+k*(alpha-1));

end
model.predict = @predict;
model.sample = @sample;
model.mu = mu;
model.pic = pic;
model.r = r;
model.L = L;

end

function samples = sample(model,t)

[k,d] = size(model.mu);

samples = zeros(t,d);
for i = 1:t
    k =sampleDiscrete(model.pic);
    samples(i,:) = rand(1,d) < model.mu(k,:);
    end
end
