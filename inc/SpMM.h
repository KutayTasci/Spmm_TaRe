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

void map_csr(SparseMat *A, TP_Comm *comm);
void spmm_tp(SparseMat* A, Matrix* B, Matrix* C, TP_Comm* comm);

#endif //SPMM_TARE_SPMM_H
