clear all;
load mnist
%Xtrain = Xtrain(1:1000,:);
alpha = 1;
beta = alpha;
model = densityBernoulli(Xtrain,alpha,beta);

nlls = model.predict(model,Xtest);
averageNLL = sum(nlls)/size(Xtest,1)

samples = model.sample(model,4);
figure(1);
for i = 1:4
    subplot(2,2,i);
    imagesc(reshape(samples(i,:),[28 28])');
end

