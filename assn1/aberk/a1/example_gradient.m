close all; clear all;
load binaryLinear.mat

lambda = 1;
% S 3.1
method = {'hermite', 'boydvandenberghe', 'barzilaiborwein', 'lipschitz', 'newton'};

for j = 1:length(method)
    display(method{j});
    tic;
    model = logisticL2(X,y,lambda, method{j});
    toc;
end

binaryClassifier2Dplot(X,y,model);


%%
close all; clear all;
load rcv1_train.binary.mat

lambda = 1;
% S 3.2
method = {'hfn'};

for j = 1:length(method)
    display(method{j});
    tic;
    model = logisticL2(X,y,lambda, method{j});
    toc;
end