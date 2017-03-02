clear all;
load SSL.mat

nClasses = numel(unique(y));

model = KNN(X, y, 1);
yhat = model.predict(model, Xtest);
fprintf('KNN accuracy is %.2f\n', mean(yhat==ytest));

model_g = generativeGaussian(X, y,nClasses);
yhat = model_g.predict(model_g, Xtest);
fprintf('Gaussian Gen. Model. accuracy is %.2f\n', mean(yhat==ytest));

 model_ssl = generativeGaussianSSL(X, y, Xtilde,nClasses,12,0);
 yhat = model_ssl.predict(model_ssl, Xtest);
 fprintf('SSL Gauss. Gen. Model. Soft EM accuracy is %.2f\n', mean(yhat==ytest));
 
 model_ssl = generativeGaussianSSL(X, y, Xtilde,nClasses,12,1);
 yhat = model_ssl.predict(model_ssl, Xtest);
 fprintf('SSL Gauss. Gen. Model. Hard EM accuracy is %.2f\n', mean(yhat==ytest));
