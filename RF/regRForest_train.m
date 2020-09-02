function rforest = regRForest_train(X, Y, params)
% Grow a random forest

%% Pre-processing 
ndata = size(X, 1);
ntree = params.ntree;
depth_tree = params.depth_tree;
ndata_tree = params.ndata_tree;
mtry = params.mtry;
thd_ndsize = params.thd_ndsize;
if 1 > (thd_ndsize-4)
    max_mnode = 2*ndata + 1;
else
    max_mnode = 2*floor(ndata/(thd_ndsize-4)) + 1;
end
trees = cell(ntree, 1);

%% Grow trees
for i = 1:ntree
    % prepare the training data for each tree, smapling with replacement
    idata_tree = randi(ndata, ndata_tree, 1);
    tree = mextree(single(X), single(Y), int32(idata_tree), int32(mtry), int32(max_mnode), int32(depth_tree), int32(thd_ndsize));
    tree = todouble_tree(tree);
    trees{i} = tree;
end
rforest.trees = trees;
rforest.ndims_pred = size(Y, 2);

end

function tree = todouble_tree(tree)

tree.dim_fea = double(tree.dim_fea);
tree.thd_fea = double(tree.thd_fea);
tree.isleafnode = double(tree.isleafnode);
tree.idx_cnd = double(tree.idx_cnd);
tree.depth_node = double(tree.depth_node);
tree.pred_node = double(tree.pred_node);

end