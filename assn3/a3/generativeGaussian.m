function model = generativeGaussian(Xtrain,Ytrain,k)

[n,d] = size(Xtrain);

%training dataset
model.Xtrain = Xtrain;
model.Ytrain = Ytrain;
%number of class
model.K = k;

%Initialize cell array to store 
%averages mu, 
%probability of each class theta
%Covariances matrices SIGMA
mu = cell(k,1);
theta = zeros(k,1);
SIGMA = cell(k,1);

%For each class i
for i=1:k
    
    ind = find(Ytrain == i);
    %Compute probability of the class theta_i
    theta(i) = sum(Ytrain == i)/n;
    %Compute average mu_i
    mu{i} = (1/(n*theta(i)))*sum(Xtrain(ind,:))';
    
    %Compute cavariance matrix SIGMA_i
    SIGMA{i} = (1/(n*theta(i)))*((Xtrain(ind,:)'-mu{i})*(Xtrain(ind,:)'-mu{i})');
    
    
end
model.predict = @(model, Xtest) predict(model, Xtest);
model.mu = mu;
model.theta = theta;
model.SIGMA = SIGMA;

end

%Predict function for test data
function yhat = predict(model, Xtest)

[nTest,d] = size(Xtest);

%Probability distribution
ytemp = zeros(nTest, model.K);

%for each test data
for i=1:nTest
    %compute the probability to be in each class
    for j=1:model.K
        ytemp(i,j) = mvnpdf(Xtest(i,:)',model.mu{j},model.SIGMA{j});
    end
end
%Classify in the class with maximum probability
[~,yhat] = max(ytemp,[],2);

end
