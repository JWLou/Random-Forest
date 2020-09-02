function [fea_rf, pred_rf] = regRForest_predict(X, rforest)
% Predict random forest features. Each dimension of the feature is a binary value
% indicating what leafnodes the sample has fallen into.

ntree = length(rforest.trees);
fea_rf = cell(1, ntree);
pred_rf = zeros(ntree, rforest.ndims_pred);
for i = 1:ntree
    [fea_tree, pred_tree] = regTree_predict(X, rforest.trees{i});
    fea_rf{i} = fea_tree;
    pred_rf(i, :) = pred_tree;
end
fea_rf = cell2mat(fea_rf);
pred_rf = mean(pred_rf);
    
end