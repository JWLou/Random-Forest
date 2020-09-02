function [fea_tree, pred_tree] = regTree_predict(X, tree)
% Extract tree features and the prediction. Each dimension of the feature is a binary value
% indicating what leafnodes the sample has fallen into.

fea_tree = zeros(1, sum(tree.isleafnode));
idx_nd = 1;
while 1
    if tree.isleafnode(idx_nd)
        break;
    end
    fea = X(tree.dim_fea(idx_nd));
    if fea < tree.thd_fea(idx_nd)
       idx_nd = tree.idx_cnd(idx_nd, 1);
    else
       idx_nd = tree.idx_cnd(idx_nd, 2);
    end
end
idx_leafnode = find(tree.isleafnode);
fea_tree(idx_leafnode==idx_nd) = 1;
pred_tree = tree.pred_node(idx_nd, :);

end