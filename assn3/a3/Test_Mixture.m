n = 100;
d = 3;
k = 10;
X = round(rand(n,d));
alpha = 2;
beta = 2;
its = 10;
model = MixtureBernoulli(X,alpha,1,k,its)