function params = paramsRForest(X, ntree, dtree)
% Parameters of the random forest

switch nargin
    case 3
        params.ntree = ntree;
        params.depth_tree = dtree;
    case 2
        params.ntree = ntree;
        params.depth_tree = 20;
    otherwise
        params.ntree = 100;
        params.depth_tree = 20;
end
params.ndata_tree = size(X,1); 
params.mtry = max(floor(sqrt(size(X,2))),1); % max(floor(sqrt(size(X,2))),1)
params.thd_ndsize = 5;

end