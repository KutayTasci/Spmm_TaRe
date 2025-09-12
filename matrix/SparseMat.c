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
        A->nnz = 0;

        FILE* fpmat = fopen(fName, "rb");
        int idx_size;
        fread(&idx_size, sizeof(int), 1, fpmat);
        fread(&(A->gm), sizeof(int), 1, fpmat);
        fread(&(A->gn), sizeof(int), 1, fpmat);

        fseek(fpmat, 3 * sizeof(int) + (world_rank * sizeof(int64_t)), SEEK_SET);
        fread(&sloc, sizeof(int64_t), 1, fpmat);

        fseek(fpmat, sloc, SEEK_SET);
        fread(&(A->m), sizeof(int), 1, fpmat);
#ifndef  use_i64
        if (idx_size == 8) {
            long long nnz_temp;
            fread(&nnz_temp, sizeof(long long), 1, fpmat);
            A->nnz = (idx_t)nnz_temp;
        }
        else
#endif
        fread(&(A->nnz), idx_size, 1, fpmat);

        A->ia = (idx_t*)malloc(sizeof(idx_t) * (A->m + 1));
        A->ja = (int*)malloc(sizeof(int) * A->nnz);
        A->ja_mapped = malloc(sizeof(int) * A->nnz);
        A->val = (double*)malloc(sizeof(double) * A->nnz);

#ifdef use_i64
        if (idx_size == 4) {
            int* ia_temp = malloc(idx_size * (A->m + 1));
#else
        if (idx_size == 8) {
            long long* ia_temp = malloc(idx_size * (A->m + 1));
#endif
            fread(ia_temp, idx_size, A->m + 1, fpmat);
            for (int i = 0; i < A->m + 1; ++i) {
                A->ia[i] = (idx_t)ia_temp[i];
            }
            free(ia_temp);
        }
        else {
            fread(A->ia, idx_size, A->m + 1, fpmat); // file and memory should match
        }

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

