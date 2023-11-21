//
// Created by kutay on 20.11.2023.
//
#include "../inc/DenseMat.h"
#include <stdlib.h>

/*
 * Allocate matrix object with memory mapping
 * Added by @Kutay
*/
Matrix* matrix_create(int row, int col, int gm, int gn) {
    Matrix *matrix =(Matrix *)  malloc(sizeof(Matrix));
    matrix->m = row;
    matrix->n = col;
    matrix->gm = gm;
    matrix->gn = gn;
    double* data;
    int total = row * col;

    data = (double *) malloc(total * sizeof(double));
    matrix->entries = (double **) malloc(row * sizeof(double*));

    for (int i = 0; i < row; i++) {
        matrix->entries[i] = &(data[col*i]);
        //memset(matrix->entries[i], 0, col * sizeof(double));
    }
    return matrix;
}

/*
 * Free Matrix
 * Added by @Kutay
*/
void matrix_free(Matrix *m) {
    free(m->entries[0]);
    free(m->entries);
    free(m);
    m = NULL;
}

/*
 * Fills a given matrix with value
 * Fills random when given NULL
 * Added by @Kutay
*/
void matrix_fill_double(Matrix *m, double num) {
    int i, j;
    if (num) {
        for (i = 0; i < m->m; i++) {
            for (j = 0; j < m->n; j++) {
                m->entries[i][j] = num;
            }
        }
    } else {
        for (i = 0; i < m->m; i++) {
            for (j = 0; j < m->n; j++) {
                num = rand();
                m->entries[i][j] = num;
            }
        }
    }
}