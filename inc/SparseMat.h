//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_SPARSEMAT_H
#define SPMM_TARE_SPARSEMAT_H

#define STORE_BY_COLUMNS 0
#define STORE_BY_ROWS    1

#include "mkl.h"

/*
 * Sparse matrix data structure added by @Kutay
 */
typedef struct {
    int *ia;    // rows of A in csr format
    int *ja_mapped;
    int *ja;    // cols of A in csr format
    double *val; // values of A in csr format

    int m;
    int n;
    int nnz;
    int gm, gn;
    int store;

    int *l2gMap;
    int *inPart;
    
    sparse_matrix_t BLAS_A;
    
    
} SparseMat;

SparseMat* readSparseMat(char* fName, int partScheme, char* inPartFile);
void sparseMatFree(SparseMat* A);

#endif //SPMM_TARE_SPARSEMAT_H
