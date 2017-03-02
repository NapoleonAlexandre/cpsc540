n = 100; k = 10; d = 784;
rp = randperm(60000);
X = Xtrain(rp(1:n),:);
model = MixtureBernoulli(X,2,1,10,1);
%model2 = MixtureBernoulli(X,2,1,10,2);
%model3 = MixtureBernoulli(X,2,10,3);
%model4 = MixtureBernoulli(X,2,10,4);
%model5 = MixtureBernoulli(X,2,10,5);


%%
samples = model.sample(model,4);
figure(1);
for i = 1:4
    subplot(2,2,i);
    imagesc(reshape(samples(i,:),[28 28])');
end

%%
figure(2);
k = [1 2 9 4]
for i = 1:4
    subplot(2,2,i);
    imagesc(reshape(model.mu(k(i),:),[28 28])');
end