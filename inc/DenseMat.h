//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_DENSEMAT_H
#define SPMM_TARE_DENSEMAT_H


typedef struct {
    double** entries;
    int m;
    int n;
    int gm;
    int gn;
} Matrix;

Matrix* matrix_create(int row, int col, int gm, int gn);
void matrix_free(Matrix *m);

void matrix_fill_int(Matrix *m, int num);
void matrix_fill_double(Matrix *m, double num);

#endif //SPMM_TARE_DENSEMAT_H
