function model = generativeStudent(Xtrain,Ytrain,k)

[n,d] = size(Xtrain);

%our model is a set of k multivariate T model
model.tdist = cell(k,1);
%For each class
for i=1:k
    %Fit a multivariate T model
    ind = find(Ytrain == i);
    model.tdist{i} = multivariateT(Xtrain(ind,:));
    model.tdist{i}.Xtrain = Xtrain(ind,:);
    model.tdist{i}.Ytrain = Ytrain(ind);
end
model.predict = @(model, Xtest) predict(model, Xtest);
model.K = k;
end

%Predict function for test data
function yhat = predict(model, Xtest)
    
[nTest,d] = size(Xtest);

ytemp = zeros(nTest, model.K);

%for each test data
for i=1:nTest
    %compute the probability to be in each class
    for j=1:model.K
        ytemp(i,j) = model.tdist{j}.pdf(model.tdist{j},Xtest(i,:));
    end
end
%Classify in the class with maximum probability
[~,yhat] = max(ytemp,[],2);

end
