function testFriedman

% load a dataset (WARNING: the dataset must be converted in single format)
data = load('friedman1.csv');
X = data(:,1:10); % input
Y = data(:,11); % output
idxtrain = 1:300;
idxtest = 8001:10000;

% train
params_rf = paramsRForest(X(idxtrain, :));    
rforest = regRForest_train(X(idxtrain, :), Y(idxtrain), params_rf);

% test
ntest = length(idxtest);
pred = zeros(ntest, size(Y,2));
for i = 1:ntest
    [~, pred(i,:)] = regRForest_predict(X(idxtest(i), 1:10), rforest);    
end

% error
mean((pred-Y(idxtest)).^2)

end

