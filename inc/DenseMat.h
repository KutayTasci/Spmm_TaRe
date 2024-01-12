//
// Created by kutay on 20.11.2023.
//

#ifndef SPMM_TARE_DENSEMAT_H
#define SPMM_TARE_DENSEMAT_H

#include "../inc/CommHandler.h"

typedef struct {
    double** entries;
    int m;
    int lcl_m;
    int n;
    int gm;
    int gn;
    int phase_1, phase_2;
} Matrix;

Matrix* matrix_create(int row, int col, int gm, int gn);
Matrix* matrix_create_tp(int row, int col, int gm, int gn, TP_Comm* comm);
Matrix *matrix_create_op(int row, int col, int gm, int gn, OP_Comm *comm);
void matrix_free(Matrix *m);

void matrix_fill_int(Matrix *m, int num);
void matrix_fill_double(Matrix *m, double num);

#endif //SPMM_TARE_DENSEMAT_H
