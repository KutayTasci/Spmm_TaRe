//
// Created by kutay on 20.11.2023.
//
#include "../inc/DenseMat.h"
#include <stdlib.h>
#include <mpi.h>

/*
 * Allocate matrix object with memory mapping
 * Added by @Kutay
*/
Matrix *matrix_create(int row, int col, int gm, int gn) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->m = row;
    matrix->n = col;
    matrix->gm = gm;
    matrix->gn = gn;
    double *data;
    int total = row * col;

    data = (double *) malloc(total * sizeof(double));
    matrix->entries = (double **) malloc(row * sizeof(double *));

    for (int i = 0; i < row; i++) {
        matrix->entries[i] = &(data[col * i]);
        //memset(matrix->entries[i], 0, col * sizeof(double));
    }
    return matrix;
}

Matrix *matrix_create_tp(int row, int col, int gm, int gn, TP_Comm *comm) {
    int total_row = row + comm->recvBuffer_p1.count + comm->recvBuffer_p2.count;
    Matrix *matrix = matrix_create(total_row, col, gm, gn);

    matrix->lcl_m = row;
    matrix->phase_1 = row;
    matrix->phase_2 = row + comm->recvBuffer_p1.count;
    return matrix;
}

Matrix *matrix_create_op(int row, int col, int gm, int gn, OP_Comm *comm) {
    int total_row = row + comm->recvBuffer.count;
    Matrix *matrix = matrix_create(total_row, col, gm, gn);
    matrix->lcl_m = row;
    matrix->phase_1 = row;
    return matrix;
}

/*
 * Free Matrix
 * Added by @Kutay
*/
void matrix_free(Matrix *m) {

    if (m->m > 0) {
        free(m->entries[0]);
        free(m->entries);
    }
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
        for (i = 0; i < m->lcl_m; i++) {
            for (j = 0; j < m->n; j++) {
                m->entries[i][j] = num;
            }
        }
    } else {
        for (i = 0; i < m->lcl_m; i++) {
            for (j = 0; j < m->n; j++) {
                num = rand();
                m->entries[i][j] = num;
            }
        }
    }
}


void map_comm_tp(TP_Comm *Comm, Matrix *B) {

    int i, part;
    udx_t base, range;
    for (i = 0; i < Comm->msgRecvCount_p1; i++) {
        part = Comm->recv_proc_list_p1[i];
        range = Comm->recvBuffer_p1.proc_map[part + 1] - Comm->recvBuffer_p1.proc_map[part];
        base = B->phase_1 + Comm->recvBuffer_p1.proc_map[part];

        MPI_Recv_init(&(B->entries[base][0]),
                      range * B->n,
                      MPI_DOUBLE,
                      part,
                      0,
                      MPI_COMM_WORLD,
                      &(Comm->recv_ls_p1[i]));
    }


    for (i = 0; i < Comm->msgRecvCount_p2; i++) {
        part = Comm->recv_proc_list_p2[i];
        range = Comm->recvBuffer_p2.proc_map[part + 1] - Comm->recvBuffer_p2.proc_map[part];
        base = B->phase_2 + Comm->recvBuffer_p2.proc_map[part];

        MPI_Recv_init(&(B->entries[base][0]),
                      range * B->n,
                      MPI_DOUBLE,
                      part,
                      1,
                      MPI_COMM_WORLD,
                      &(Comm->recv_ls_p2[i]));


    }

}

void map_comm_op(OP_Comm *comm, Matrix *B) {
    int i, part;
    udx_t base, range;

    for (i = 0; i < comm->msgRecvCount; i++) {
        part = comm->recv_proc_list[i];
        range = comm->recvBuffer.proc_map[part + 1] - comm->recvBuffer.proc_map[part];
        base = B->phase_1 + comm->recvBuffer.proc_map[part];

        MPI_Recv_init(&(B->entries[base][0]),
                      range * B->n,
                      MPI_DOUBLE,
                      part,
                      0,
                      MPI_COMM_WORLD,
                      &(comm->recv_ls[i]));

    }
}

