clear all;
load SSL.mat

nClasses = numel(unique(y));

it =15;
acc0 = zeros(it,1);
acc1 = zeros(it,1);

for i=1:it
 model = generativeGaussianSSL(X, y, Xtilde,nClasses,i,0);
 yhat = model.predict(model, Xtest);
 acc0(i) = mean(yhat==ytest);
 
 model = generativeGaussianSSL(X, y, Xtilde,nClasses,i,1);
 yhat = model.predict(model, Xtest);
 acc1(i) = mean(yhat==ytest);

end

figure(1);
plot(acc0);
hold on;
plot(acc1);
legend('Soft','Hard');
xlabel('# iterations');
ylabel('Accuracy %');
