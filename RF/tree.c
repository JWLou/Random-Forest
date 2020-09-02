/*
 * tree.c
 * Jianwen Lou, 29/10/2019
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// input variables
float *X = NULL;              
float *Y = NULL;
int *idataset_tree = NULL;
int mtry;
int max_mnode;
int depth_tree;
int thd_ndsize;

// output variables
float *thd_fea = NULL;    // feature threshold for node split 
int *dim_fea = NULL;      // feature dimension for node split 
int *tosplit = NULL;      // indicate if the node should be split
int *isleafnode = NULL;	  // indicate if the node is a leaf node
int *depth_node = NULL;   // the layer of the node in the tree
int **inode_chd = NULL;   // the indices of child nodes
float **pred_node = NULL; // the prediction of a node 

// intermediate variables
int ndata_tree;
int ndims_pred;
int ndims_fea;
float thd_split, thd_curr;
int dim_split;
int idata_split;
double valq_max, valq_curr; // value of the quality function for finding the optimal split 
double var_node;
double **var_table = NULL;  // a 3 X (2*ndims_pred+1) array for calculating the variances of the node and its two child nodes. 
int **idata_node = NULL;    // the indices of data fed into the node
int *idataset_node = NULL;
int *dimset_fea = NULL;

// get a random integer
int getrand_int(int maxval) {
  return (int)floor((double)rand()*maxval*1.0/(RAND_MAX+1.0));
}

// allocate and free memory for multi-dimensional array
double **malloc_array_double(int nrow, int ncol)
{
	double **myarray;
	int i, j;

	myarray = (double **) malloc(nrow * sizeof(double *));
	if (myarray == NULL)
		return NULL;
	for (i = 0; i < nrow; i++)
	{
		myarray[i] = (double *) malloc(ncol * sizeof(double));
		if(myarray[i] == NULL)
		{
			for (j = 0; j < i; j++)
				free((double *) myarray[j]);
			return NULL;
		}
	}
	return myarray;
}

void mfree_array_double(double **myarray, int nrow)
{
	int i;
	if (myarray != NULL)
	{
		for (i = 0; i < nrow; i++)
			free((double *) myarray[i]);
		free((double **) myarray);
	}
}

float **malloc_array_float(int nrow, int ncol)
{
	float **myarray;
	int i, j;

	myarray = (float **) malloc(nrow * sizeof(float *));
	if (myarray == NULL)
		return NULL;
	for (i = 0; i < nrow; i++)
	{
		myarray[i] = (float *) malloc(ncol * sizeof(float));
		if(myarray[i] == NULL)
		{
			for (j = 0; j < i; j++)
				free((float *) myarray[j]);
			return NULL;
		}
	}
	return myarray;
}

void mfree_array_float(float **myarray, int nrow)
{
	int i;
	if (myarray != NULL)
	{
		for (i = 0; i < nrow; i++)
			free((float *) myarray[i]);
		free((float **) myarray);
	}
}

int **malloc_array_int(int nrow, int ncol)
{
	int **myarray;
	int i, j;

	myarray = (int **) malloc(nrow * sizeof(int *));
	if (myarray == NULL)
		return NULL;
	for (i = 0; i < nrow; i++)
	{
		myarray[i] = (int *) malloc(ncol * sizeof(int));
		if(myarray[i] == NULL)
		{
			for (j = 0; j < i; j++)
				free((int *) myarray[j]);
			return NULL;
		}
	}
	return myarray;
}

void mfree_array_int(int **myarray, int nrow)
{
	int i;
	if (myarray != NULL)
	{
		for (i = 0; i < nrow; i++)
			free((int *) myarray[i]);
		free((int **) myarray);
	}
}

// free tree memory
void mfree_tree()
{
	if (thd_fea != NULL) {free((float *) thd_fea); thd_fea = NULL;}
	if (dim_fea != NULL) {free((int *) dim_fea); dim_fea = NULL;}
	if (tosplit != NULL) {free((int *) tosplit); tosplit = NULL;}
	if (isleafnode != NULL) {free((int *) isleafnode); isleafnode = NULL;}
	if (depth_node != NULL) {free((int *) depth_node); depth_node = NULL;}
	if (inode_chd != NULL) {mfree_array_int(inode_chd, max_mnode); inode_chd = NULL;}
	if (pred_node != NULL) {mfree_array_float(pred_node, max_mnode); pred_node = NULL;}
	if (idata_node != NULL) {mfree_array_int(idata_node, max_mnode); idata_node = NULL;}
	if(var_table != NULL) {mfree_array_double(var_table, 3); var_table = NULL;}
	if(idataset_tree != NULL) {free((int *) idataset_tree); idataset_tree = NULL;}
	if(idataset_node != NULL) {free((int *) idataset_node); idataset_node = NULL;}
	if(dimset_fea != NULL) {free((int *) dimset_fea); dimset_fea = NULL;}
}

// allocate tree memory 
int malloc_tree()
{
	thd_fea = (float *) malloc(max_mnode * sizeof(float)); 
	if (thd_fea == NULL) return 0;
	dim_fea = (int *) malloc(max_mnode * sizeof(int));
	if (dim_fea == NULL) {mfree_tree(); return 0;}
	tosplit = (int *) malloc(max_mnode * sizeof(int));
	if (tosplit == NULL) {mfree_tree(); return 0;}
	isleafnode = (int *) malloc(max_mnode * sizeof(int));
	if (isleafnode == NULL) {mfree_tree(); return 0;}
	depth_node = (int *) malloc(max_mnode * sizeof(int));
	if (depth_node == NULL) {mfree_tree(); return 0;}
	inode_chd = malloc_array_int(max_mnode, 2);
	if (inode_chd == NULL) {mfree_tree(); return 0;}
	pred_node = malloc_array_float(max_mnode, ndims_pred);
	if (pred_node == NULL) {mfree_tree(); return 0;}
	idata_node = malloc_array_int(max_mnode, 2);
	if (idata_node == NULL) {mfree_tree(); return 0;}
	var_table = malloc_array_double(3, 2*ndims_pred+1);
	if (var_table == NULL) {mfree_tree(); return 0;}
	idataset_tree = (int *) malloc(ndata_tree * sizeof(int));
	if (idataset_tree == NULL) {mfree_tree(); return 0;}
	idataset_node = (int *) malloc(ndata_tree * sizeof(int));
	if (idataset_node == NULL) {mfree_tree(); return 0;}
	dimset_fea = (int *) malloc(ndims_fea * sizeof(int));
	if (dimset_fea == NULL) {mfree_tree(); return 0;}

	return 1;
}

// get the feature value
float get_feaval(int irow, int icol) 
{
  return (float) X[icol*ndata_tree+irow];
}

// get the regression target value
float get_regtar(int irow, int icol)
{
	return (float) Y[icol*ndata_tree+irow];
}

// calculate the prediction vector for the leaf node
void cal_leafpred(int *idataset, int idata_start, int idata_end, int k)
{
	int i, j, ndata;

	ndata = idata_end - idata_start + 1;
	for (j = 0; j < ndims_pred; j++)
	{
		for (i = idata_start; i <= idata_end; i++)		
		{
			float y = get_regtar(idataset_tree[idataset[i]], j);
			pred_node[k][j] += y;
		}
		pred_node[k][j] = (float) (pred_node[k][j]/ndata);
	}
}

// calculate the variance reduction of the split
double cal_valq()
{
	int i;
	double var_lchd, var_rchd;
	var_lchd = 0.0;
	var_rchd = 0.0;

	var_table[2][0] = var_table[0][0] - var_table[1][0];
	for (i = 0; i < ndims_pred; i++)
	{
		// i=0 parent node; i=1 left child node; i=2 right child node
		var_table[2][2*i+1] = var_table[0][2*i+1] - var_table[1][2*i+1];
		var_table[2][2*i+2] = var_table[0][2*i+2] - var_table[1][2*i+2];

		var_lchd += fabs(var_table[1][2*i+2] - var_table[1][2*i+1]*var_table[1][2*i+1]/var_table[1][0]);
		var_rchd += fabs(var_table[2][2*i+2] - var_table[2][2*i+1]*var_table[2][2*i+1]/var_table[2][0]);
	}

	return ((var_node-var_lchd-var_rchd)/var_node);
}

// sort the feature-dim values in ascending order using the Quicksort algorithm
#define QUICKSORT_STACK_SIZE 50
#define SWAP(a,b) temp=(a); a=(b);(b)=temp
#define VAL(idx) get_feaval(idataset_tree[idx], dim)
void sort_data(int *idataset, int idata_start, int idata_end, int dim)
{
	int i, j;
	int lo = idata_start, hi = idata_end, mid;
	int stack_qs[QUICKSORT_STACK_SIZE];
	int stack_top = -1;
	int pivot_idx;
	float pivot_val;
	int temp;

	for(;;)
	{
		if(hi-lo < 7)
		{
			// perform insertion sort when the array size is small
			for (i = lo+1; i <= hi; i++)
			{
				pivot_idx = idataset[i];
				pivot_val = VAL(pivot_idx);
				for (j = i-1; (j>=lo) && (VAL(idataset[j])>pivot_val); j--)
					idataset[j+1] = idataset[j];
				idataset[j+1] = pivot_idx;
			}
			if (stack_top == -1)
				break;
			hi = stack_qs[stack_top--];
			lo = stack_qs[stack_top--];
		}
		else
		{
			// median-of-three, make VAL(idataset[mid]) <= VAL(idataset[lo]) <= VAL(idataset[hi])
			mid = lo + ((hi - lo) >> 1);
			if (VAL(idataset[mid]) > VAL(idataset[hi]))
			{
				SWAP(idataset[mid], idataset[hi]);
			}
			if (VAL(idataset[lo]) > VAL(idataset[hi]))
			{
				SWAP(idataset[lo],idataset[hi]);
			}
			if (VAL(idataset[mid]) > VAL(idataset[lo])) 
			{
				SWAP(idataset[mid], idataset[lo]);
			}
			
			// one partition 
			pivot_idx = idataset[lo];
			pivot_val = VAL(pivot_idx);
			i = lo;
			j = hi + 1;
			for (;;)
			{					
				while (VAL(idataset[++i]) < pivot_val && i < hi);
				while (VAL(idataset[--j]) > pivot_val);
				if (i >= j) break;
				SWAP(idataset[i], idataset[j]);
			}
			idataset[lo] = idataset[j];
			idataset[j] = pivot_idx;

			// preserve the info of sub-arrays in stack
			stack_top += 2;
			if (stack_top > QUICKSORT_STACK_SIZE)
				return;
			if (hi-i+1 >= j-lo)
			{
				stack_qs[stack_top] = hi;
				stack_qs[stack_top-1] = i;
				hi = j - 1;
			}
			else
			{
				stack_qs[stack_top] = j - 1;
				stack_qs[stack_top-1] = lo;
				lo = i;
			}
		}
	}
}

// partition the idataset with the found feature dimension and threshold
int partition_idataset(int *idataset, int idata_start, int idata_end, int dim, float thd)
{	
	int i, j;
	int lo = idata_start, hi = idata_end;
	int temp;
	i = lo - 1;
	j = hi + 1;
	for (;;)
	{					
		while (VAL(idataset[++i]) < thd && i < hi);
		while (VAL(idataset[--j]) >= thd && j > lo);
		if (i >= j) break;
		SWAP(idataset[i], idataset[j]);
	}

	return j; 
}

// find the best threshold for node splitting 
void find_thd(int *idataset, int idata_start, int idata_end, int dim)
{
	float feaval_prev, feaval_curr;
	double valqdim_split = -1.0, valqdim_curr = -1.0;
	float thddim_split;
	int i, j;

	// initialize the var_table for the left child node
	var_table[1][0] = 0.0;
	for (j = 0; j < ndims_pred; j++)
	{
		var_table[1][2*j+1] = 0.0;
		var_table[1][2*j+2] = 0.0;
	}

	// sort the index set in ascending order
	sort_data(idataset, idata_start, idata_end, dim);

	// go through all the values along the current feaure dimension
	feaval_prev = get_feaval(idataset_tree[idataset[idata_start]], dim);
	for (i = idata_start; i < idata_end; i++)
	{
		var_table[1][0] += 1;
		for (j = 0; j < ndims_pred; j++)
		{
			float y = get_regtar(idataset_tree[idataset[i]], j);
			var_table[1][2*j+1] += y;
			var_table[1][2*j+2] += y*y;
		}

		feaval_curr = get_feaval(idataset_tree[idataset[i+1]], dim);
		if(feaval_curr != feaval_prev)
		{
			valqdim_curr = cal_valq();
			if (valqdim_curr > valqdim_split)
			{
				valqdim_split = valqdim_curr;
				thddim_split = (feaval_curr + feaval_prev)/2.0;
				if (feaval_prev >= thddim_split)
					thddim_split = feaval_curr;
			}
			feaval_prev = feaval_curr;
		}
	}

	if (valqdim_split >= 0.0)
	{
		thd_curr = thddim_split;
		valq_curr = valqdim_split;
	}
}

// split a node 
void split_node(int *idataset, int idata_start, int idata_end)
{
	int i, j;
	int dim_remain = ndims_fea;
	int dim_rand, tmp;

	dim_split = -1;
    valq_max = -1.0;
	valq_curr = -1.0;
	var_node = 0.0;

	// initialise the var_table for the current node
	for (i = 0; i < 2*ndims_pred+1; i++)
		var_table[0][i] = 0.0;
	for(i = idata_start; i <= idata_end; i++)
	{
		var_table[0][0] += 1;
		for(j = 0; j < ndims_pred; j++)
		{
			float y = get_regtar(idataset_tree[idataset[i]], j);
			var_table[0][2*j+1] += y;
			var_table[0][2*j+2] += y*y;
		}
	}
	for (i = 0; i < ndims_pred; i++)
		var_node += fabs(var_table[0][2*i+2] - var_table[0][2*i+1]*var_table[0][2*i+1]/var_table[0][0]);

	// test mtry random feature dimensions
	for (i = 0; i < ndims_fea; i++)
		dimset_fea[i] = i;
	for (i = 0; (i<mtry) && (dim_remain!=0); i++)
	{
		dim_rand = getrand_int(dim_remain);
		find_thd(idataset, idata_start, idata_end, dimset_fea[dim_rand]);
		if (valq_curr > valq_max)
		{	
			valq_max = valq_curr;		
			thd_split = thd_curr;
			dim_split = dimset_fea[dim_rand];
		}

		// swap feature dimensions
		dim_remain--;
		if (dim_remain != 0)
		{
			tmp = dimset_fea[dim_rand];
			dimset_fea[dim_rand] = dimset_fea[dim_remain];
			dimset_fea[dim_remain] = tmp;
		}
	}
}

// initialise the tree
void init_tree(int *idataset_tree_matlab)
{
	int i, j;
	
	for (i = 0; i < max_mnode; i++)
	{
		thd_fea[i] = 0.0;
		dim_fea[i] = -1;
		tosplit[i] = 0;
		isleafnode[i] = 0;
		depth_node[i] = 0;
		inode_chd[i][0] = -1; inode_chd[i][1] = -1;
		idata_node[i][0] = -1; idata_node[i][1] = -1;
		for (j = 0; j < ndims_pred; j++)
			pred_node[i][j] = 0.0;
	}
	for (i = 0; i < ndata_tree; i++)
		idataset_tree[i] = idataset_tree_matlab[i] - 1;
	for (i = 0; i < ndata_tree; i++)
		idataset_node[i] = i;
	for (i = 0; i < ndims_fea; i++)
		dimset_fea[i] = i;
}

// build a tree
void build_tree(int *idataset_tree_matlab)
{
    int ndata_lchd, ndata_rchd;
	int inode_lchd, inode_rchd;
	int ke = 0; // the index of the last added node
	int k;
    
	// initialization
	init_tree(idataset_tree_matlab);

	tosplit[0] = 1;
	depth_node[0] = 1;
	idata_node[0][0] = 0; idata_node[0][1] = ndata_tree - 1; // start and end indices indicating the data samples to feed the current node
	
	// loop through the nodes
	for (k = 0; k < max_mnode; k++)
	{
		// break the loop if the number of nodes added has reached its maximum amount
		if (k > ke || ke >= (max_mnode-1))
			break;

		// skip if the node is not to be split
		if (!tosplit[k])
			continue;

		// split a node
		split_node(idataset_node, idata_node[k][0], idata_node[k][1]);
		if (dim_split == -1)
		{
			// split failed
			isleafnode[k] = 1;
			cal_leafpred(idataset_node, idata_node[k][0], idata_node[k][1], k);
			continue;
		}
		thd_fea[k] = thd_split;
		dim_fea[k] = dim_split;		
		idata_split = partition_idataset(idataset_node, idata_node[k][0], idata_node[k][1], dim_fea[k], thd_fea[k]);
		inode_lchd = ke + 1; // left child node index
		inode_chd[k][0] = inode_lchd;
		inode_rchd = ke + 2; // right child node index
		inode_chd[k][1] = inode_rchd;

		// for left child node
		idata_node[inode_lchd][0] = idata_node[k][0];
		idata_node[inode_lchd][1] = idata_split;
		depth_node[inode_lchd] = depth_node[k] + 1;
		ndata_lchd = idata_split - idata_node[k][0] + 1;
		if (ndata_lchd <= thd_ndsize || depth_node[inode_lchd] >= depth_tree)
		{
			isleafnode[inode_lchd] = 1;
			cal_leafpred(idataset_node, idata_node[inode_lchd][0], idata_node[inode_lchd][1], inode_lchd);
		}
		else
			tosplit[inode_lchd] = 1;

		// for right child node
		idata_node[inode_rchd][0] = idata_split + 1;
		idata_node[inode_rchd][1] = idata_node[k][1];
		depth_node[inode_rchd] = depth_node[k] + 1;
		ndata_rchd = idata_node[k][1] - idata_split;
		if (ndata_rchd <= thd_ndsize || depth_node[inode_rchd] >= depth_tree)
		{
			isleafnode[inode_rchd] = 1;
			cal_leafpred(idataset_node, idata_node[inode_rchd][0], idata_node[inode_rchd][1], inode_rchd);
		}
		else
			tosplit[inode_rchd] = 1;

		// augment the tree by two nodes
		ke = ke + 2;
	}
}

