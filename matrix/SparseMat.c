//
// Created by kutay on 20.11.2023.
//
#include "../inc/SparseMat.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * Reading CSR Matrix in parallel
 * Handles memory allocation
 * File format follows the output format of "Util" code by Oguz Selvitopi
 * Added by @Kutay
*/
SparseMat* readSparseMat(char* fName, int partScheme, char* inPartFile) {
    if (partScheme == STORE_BY_COLUMNS) {
        printf("STORE_BY_COLUMNS not implemented.");
        exit(EXIT_FAILURE);
    }
    else {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int64_t sloc;

        SparseMat* A = (SparseMat*)malloc(sizeof(SparseMat));

        FILE* fpmat = fopen(fName, "rb");

        fread(&(A->gm), sizeof(int), 1, fpmat);
        fread(&(A->gn), sizeof(int), 1, fpmat);

        fseek(fpmat, 2 * sizeof(int) + (world_rank * sizeof(int64_t)), SEEK_SET);
        fread(&sloc, sizeof(int64_t), 1, fpmat);

        fseek(fpmat, sloc, SEEK_SET);
        fread(&(A->m), sizeof(int), 1, fpmat);
        fread(&(A->nnz), sizeof(int), 1, fpmat);

        A->ia = (int*)malloc(sizeof(int) * (A->m + 1));
        A->ja = (int*)malloc(sizeof(int) * A->nnz);
        A->ja_mapped = (int*)malloc(sizeof(int) * A->nnz);
        A->val = (double*)malloc(sizeof(double) * A->nnz);

        fread(A->ia, sizeof(int), A->m + 1, fpmat);
        fread(A->ja, sizeof(int), A->nnz, fpmat);
        fread(A->val, sizeof(double), A->nnz, fpmat);

        A->store = STORE_BY_ROWS;

        A->inPart = malloc(sizeof(*(A->inPart)) * A->gn);
        A->l2gMap = malloc(sizeof(int) * A->m);

        FILE* pf = fopen(inPartFile, "rb");
        fread(A->inPart, sizeof(int), A->gn, pf);
        fclose(pf);
        int ctr = 0;
        for (int i = 0; i < A->gn; ++i) {
            if (A->inPart[i] == world_rank) {
                A->l2gMap[ctr++] = i;
            }
        }

        int* tmp = malloc(sizeof(*tmp) * A->gn);
        memset(tmp, 0, sizeof(*tmp) * A->gn);
        A->n = 0;
        for (int i = 0; i < A->m; ++i) {
            for (int j = A->ia[i]; j < A->ia[i + 1]; ++j)
                ++(tmp[A->ja[j]]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (world_rank == A->inPart[j])
                ++(tmp[j]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (tmp[j])
                ++(A->n);
        }

        free(tmp);

        fclose(fpmat);
        return A;
    }
}

/*
 * Free SparseMat Object
 * Added by @Kutay
*/
void sparseMatFree(SparseMat* A) {
    free(A->ia);
    free(A->ja);
    free(A->val);
    free(A->inPart);
    free(A->l2gMap);
    free(A);
    // A = NULL;
}

