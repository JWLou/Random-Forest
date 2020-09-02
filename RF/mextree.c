/*
 * tree.h
 * Jianwen Lou, 29/10/2019
 *
 */

#include "mex.h"
#include "tree.c"

// save the learned tree into a matlab structure
mxArray *saveStruct()
{
	mxArray *tmp;
	mxArray *treestruct;
	float *ptr_float;
	int *ptr_int;
	int i, j;
	char const* fnameset[]={"dim_fea", "thd_fea", "isleafnode", "idx_cnd", "depth_node", "pred_node"};

	treestruct = mxCreateStructMatrix(1, 1, 6, fnameset);

	tmp = mxCreateNumericMatrix(max_mnode, 1, mxINT32_CLASS, mxREAL);
	ptr_int = (int *) mxGetPr(tmp);
	for (i = 0; i < max_mnode; i++, ptr_int++)
		*ptr_int = dim_fea[i] + 1;
	mxSetField(treestruct, 0, "dim_fea", tmp);

	tmp = mxCreateNumericMatrix(max_mnode, 1, mxSINGLE_CLASS, mxREAL);
	ptr_float = (float *) mxGetPr(tmp);
	for (i = 0; i < max_mnode; i++, ptr_float++)
		*ptr_float = thd_fea[i];
	mxSetField(treestruct, 0, "thd_fea", tmp);

	tmp = mxCreateNumericMatrix(max_mnode, 1, mxINT32_CLASS, mxREAL);
	ptr_int = (int *) mxGetPr(tmp);
	for (i = 0; i < max_mnode; i++, ptr_int++)
		*ptr_int = isleafnode[i];
	mxSetField(treestruct, 0, "isleafnode", tmp);

	tmp = mxCreateNumericMatrix(max_mnode, 2, mxINT32_CLASS, mxREAL);
	ptr_int = (int *) mxGetPr(tmp);
	for (j = 0; j < 2; j++)
		for (i = 0; i < max_mnode; i++)
		{
			*ptr_int = inode_chd[i][j] + 1;
			ptr_int++;
		}
	mxSetField(treestruct, 0, "idx_cnd", tmp);

	tmp = mxCreateNumericMatrix(max_mnode, 1, mxINT32_CLASS, mxREAL);
	ptr_int = (int *) mxGetPr(tmp);
	for (i = 0; i < max_mnode; i++, ptr_int++)
		*ptr_int = depth_node[i];
	mxSetField(treestruct, 0, "depth_node", tmp);

	tmp = mxCreateNumericMatrix(max_mnode, ndims_pred, mxSINGLE_CLASS, mxREAL);
	ptr_float = (float *) mxGetPr(tmp);
	for (j = 0; j < ndims_pred; j++)
		for (i = 0; i < max_mnode; i++)
		{
			*ptr_float = pred_node[i][j];
			ptr_float++;
		}
	mxSetField(treestruct, 0, "pred_node", tmp);

	return treestruct;
}

// mexfunction
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mxArray *xData, *yData, *idataset_learn;
	int *ptr_int;
	if (nrhs != 7)
		mexErrMsgTxt("\n The input arguments are not correct! \n");

	// parse the input arguments
	xData = (mxArray *) prhs[0];
	yData = (mxArray *) prhs[1];
	idataset_learn = (mxArray *) prhs[2];
	mtry = (int) mxGetScalar(prhs[3]);
	max_mnode = (int) mxGetScalar(prhs[4]);
	depth_tree = (int) mxGetScalar(prhs[5]);
	thd_ndsize = (int) mxGetScalar(prhs[6]);

	// initialize the tree global variables 
	X = (float *) mxGetPr(xData);
	Y = (float *) mxGetPr(yData);
	ndata_tree = mxGetM(xData);
	ndims_fea = mxGetN(xData);
	ndims_pred = mxGetN(yData);
	ptr_int = (int *) mxGetPr(idataset_learn);

	// build a tree
	if (malloc_tree())
	{
		// build tree
		build_tree(ptr_int);

		// save tree as matlab struct
		plhs[0] = saveStruct();

		// release tree memory
		mfree_tree();
	}
	else
		mexErrMsgTxt("\n Memory allocation error! \n");
}