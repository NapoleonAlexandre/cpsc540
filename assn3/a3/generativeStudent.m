function model = generativeStudent(Xtrain,Ytrain,k)

[n,d] = size(Xtrain);

model.tdist = cell(k,1);

for i=1:k
    ind = find(Ytrain == i);
    model.tdist{i} = multivariateT(Xtrain(ind,:));
    model.tdist{i}.Xtrain = Xtrain(ind,:);
    model.tdist{i}.Ytrain = Ytrain(ind);
end
model.predict = @(model, Xtest) predict(model, Xtest);
model.K = k;
end

function yhat = predict(model, Xtest)
    
[nTest,d] = size(Xtest);

ytemp = zeros(nTest, model.K);

for i=1:nTest
    for j=1:model.K
        ytemp(i,j) = model.tdist{j}.pdf(model.tdist{j},Xtest(i,:));
    end
end

[~,yhat] = max(ytemp,[],2);

end
