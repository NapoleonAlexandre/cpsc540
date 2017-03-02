function [ model ] = MixtureBernoulli(X,alpha,beta,k,its)

[n,d] = size(X);
model.predict = @predict;
model.sample = @sample;

% Initialization of the parameters \pi_c and \mu_ij
pic = ones(k,1)/k;
mu = 0.25+0.5*rand(k,d);

%Iterations
for j=1:its
    
    % E-step
    L = zeros(n,k);
    r = zeros(n,k);
    for i=1:k
        L(:,i) = sum(log(pic(i)*(mu(i,:).^X).*((1-mu(i,:)).^(1-X))),2);
    end
    
    %LogSumExp trick
    Lmax = max(max(L));
    r = L-Lmax-log(sum(exp(L-Lmax),2));
    r = exp(r);
    
    %M step
    z = sum(r);
    %Update \mu
    for i=1:k
        mu(i,:) = (sum(spdiags(r(:,i),0,n,n)*X)+beta-1)/(z(i)+k*(beta-1));
    end
    %update \pi_c
    pic = (z+alpha-1)/(n+k*(alpha-1));

end
model.mu = mu;
model.pic = pic;
model.r = r;
model.L = L;

end

function samples = sample(model,t)

[k,d] = size(model.mu);

samples = zeros(t,d);
for i = 1:t
    %Pick a random cluster
    k =sampleDiscrete(model.pic);
    samples(i,:) = rand(1,d) < model.mu(k,:);
    end
end
