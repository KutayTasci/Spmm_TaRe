//
// Created by kutay on 07.01.2024.
//

#ifndef SPMM_TARE_SPMM_H
#define SPMM_TARE_SPMM_H

#include "../inc/SparseMat.h"
#include "../inc/DenseMat.h"
#include "../inc/CommHandler.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//MODES FOR SPMM
#define WCT_FULL 10
#define WCT_PROFILE 11

typedef struct wct {
    double p1_reduce_t; // only for: tp, reduce
    double p1_comm_t;
    double post_p1_reduce_t; // only for: tp, reduce
    double p2_comm_t; // for one phase this is the total comm time
    double SpMM_t;
    double total_t;
} wct;

wct wct_init();

void wct_print(wct *wct_time);

void map_csr(SparseMat *A, TP_Comm *comm);

void map_dependencies(SparseMat *A, TP_Comm *comm);

void map_csr_op(SparseMat *A, OP_Comm *comm);

//SPMM WRAPPERS ONLY CALL THESE ON THE MAIN FUNCTION
void spmm_tp(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm, int mode, wct *wct_time);

void spmm_op(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm, int mode, wct *wct_time);

void spmm_tp_std(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm,
                 wct *wct_time); //STANDARD TP SPMM WITHOUT PARTIAL REDUCE - REQUIRES A DOUBLE POINTER FOR WCT_TIME
void spmm_tp_prf(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm,
                 wct *wct_time); //PROFILER TP SPMM WITHOUT PARTIAL REDUCE - REQUIRES AN ARRAY OF SIZE 3 FOR WCT_TIME

void spmm_tp_pr(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm,
                wct *wct_time); //STANDARD TP SPMM WITH PARTIAL REDUCE - REQUIRES A DOUBLE POINTER FOR WCT_TIME
void spmm_tp_pr_prf(SparseMat *A, Matrix *B, Matrix *C, TP_Comm *comm,
                    wct *wct_time);//PROFILER TP SPMM WITH PARTIAL REDUCE - REQUIRES AN ARRAY OF SIZE 5 FOR WCT_TIME

void spmm_op_std(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm,
                 wct *wct_time);//STANDARD OP SPMM - REQUIRES A DOUBLE POINTER FOR WCT_TIME
void spmm_op_prf(SparseMat *A, Matrix *B, Matrix *C, OP_Comm *comm,
                 wct *wct_time);//PROFILER OP SPMM - REQUIRES AN ARRAY OF SIZE 2 FOR WCT_TIME
//void spmm_reduce_op(SparseMat* A, Matrix* B, Matrix* C, OP_Comm* comm);
//void spmm_reduce_tp(SparseMat* A, Matrix* B, Matrix* C, TP_Comm* comm);

#endif //SPMM_TARE_SPMM_H
